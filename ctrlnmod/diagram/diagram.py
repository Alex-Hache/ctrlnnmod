"""The :class:`Diagram` - a block-diagram interconnection that *is itself* an
:class:`~ctrlnmod.models.ssmodels.base.SSModel`.

Because a closed-loop interconnection is a state-space system (its aggregate state
is the concatenation of the sub-blocks' states), a :class:`Diagram` exposes the
familiar ``forward(u, x, d) -> (dx, y)`` API and therefore plugs directly into the
existing integrators (:class:`~ctrlnmod.integrators.RK4Simulator`) and trainer
(:class:`~ctrlnmod.train.LitNode`) with no changes.

Wiring is described once, declaratively:

* ``blocks``       - named :class:`~ctrlnmod.blocks.base.Block` instances;
* ``connections``  - ``"src.out" -> "dst.in"`` edges between block ports;
* ``inputs``       - which block input ports are fed by the external ``u`` vector;
* ``outputs``      - which block output ports form the external ``y`` vector;
* ``disturbances`` - which block input ports are fed by the external ``d`` vector;
* ``state_taps``   - feed a dynamic block's *state* into another block's input
  (e.g. state-feedback controllers).

The dataflow is resolved by :class:`~ctrlnmod.diagram.graph.WiringGraph`, which
orders feedthrough blocks topologically and rejects algebraic loops.
"""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor

from ctrlnmod.blocks.base import Block
from ctrlnmod.models.ssmodels.base import SSModel
from ctrlnmod.diagram.graph import WiringGraph, PortRef, Connection

PortSpec = Union[str, PortRef]
ConnSpec = Union[Tuple[PortSpec, PortSpec], Connection]


def _as_ref(spec: PortSpec) -> PortRef:
    """Accept either ``"block.port"`` or ``("block", "port")``."""
    if isinstance(spec, str):
        block, port = spec.split(".")
        return (block, port)
    return (spec[0], spec[1])


class Diagram(SSModel):
    """A wired interconnection of blocks behaving as a single state-space model.

    Args:
        blocks: Mapping ``name -> Block``. The state-vector ordering follows the
            insertion order of the *dynamic* blocks in this mapping.
        connections: Iterable of ``(src, dst)`` where each endpoint is ``"block.port"``
            or ``("block", "port")``; ``src`` is an output port, ``dst`` an input port.
        inputs: Ordered list of input-port endpoints fed by the external ``u`` vector.
            Their widths are concatenated to form ``nu``.
        outputs: Ordered list of output-port endpoints whose concatenation forms ``y``
            (width ``ny``).
        disturbances: Optional ordered list of input-port endpoints fed by the external
            ``d`` vector (widths concatenated into ``nd``).
        state_taps: Optional list of ``(src_block, "dst.port")`` (or
            ``(src_block, dst_block, dst_port)``) feeding ``src_block``'s state into an
            input port of another block. The port width must equal ``src_block.nx``.
        feedthrough: Whether the diagram's own ``y`` depends directly on ``u``. Defaults
            to ``False``; set ``True`` if an external input reaches ``y`` purely through
            feedthrough blocks.
    """

    def __init__(
        self,
        blocks: Dict[str, Block],
        connections: Sequence[ConnSpec] = (),
        inputs: Sequence[PortSpec] = (),
        outputs: Sequence[PortSpec] = (),
        disturbances: Optional[Sequence[PortSpec]] = None,
        state_taps: Optional[Sequence] = None,
        feedthrough: bool = False,
    ) -> None:
        # Normalise the wiring specification ------------------------------------
        self._blocks_spec = blocks
        self._connections_spec = list(connections)
        self._inputs_spec = list(inputs)
        self._outputs_spec = list(outputs)
        self._disturbances_spec = list(disturbances) if disturbances is not None else None
        self._state_taps_spec = list(state_taps) if state_taps is not None else None

        self.input_refs: List[PortRef] = [_as_ref(s) for s in inputs]
        self.output_refs: List[PortRef] = [_as_ref(s) for s in outputs]
        self.disturbance_refs: List[PortRef] = (
            [_as_ref(s) for s in disturbances] if disturbances is not None else []
        )
        self.connections: List[Connection] = [
            (_as_ref(a), _as_ref(b)) for (a, b) in connections
        ]
        self.state_taps: List[Tuple[str, PortRef]] = []
        for tap in (state_taps or []):
            if len(tap) == 2:
                self.state_taps.append((tap[0], _as_ref(tap[1])))
            else:  # (src_block, dst_block, dst_port)
                self.state_taps.append((tap[0], (tap[1], tap[2])))

        # Dimensions ------------------------------------------------------------
        nx = sum(b.state_dim for b in blocks.values())
        nu = sum(self._port_width(r) for r in self.input_refs)
        ny = sum(self._port_width_out(r, blocks) for r in self.output_refs)
        nd = sum(self._port_width(r) for r in self.disturbance_refs)
        nd = nd if nd > 0 else None

        super().__init__(nu=nu, ny=ny, nx=nx, nd=nd, feedthrough=feedthrough)

        # Register sub-blocks as children (so .parameters()/.to() see them) -----
        self.blocks = torch.nn.ModuleDict(blocks)
        # State slicing follows insertion order of dynamic blocks.
        self._dyn_names = [name for name, b in blocks.items() if b.is_dynamic]
        self._state_slices: Dict[str, slice] = {}
        off = 0
        for name in self._dyn_names:
            w = blocks[name].state_dim
            self._state_slices[name] = slice(off, off + w)
            off += w

        # Hierarchical frame caching (same pattern as FLNSSM) -------------------
        for b in blocks.values():
            self._frame_cache.register_child(b._frame_cache)

        # Build & validate the wiring graph ------------------------------------
        seeded = list(self.input_refs) + list(self.disturbance_refs) + \
            [dst for _, dst in self.state_taps]
        self.graph = WiringGraph(
            block_ports={name: (b.in_ports, b.out_ports) for name, b in blocks.items()},
            block_feedthrough={name: b.feedthrough for name, b in blocks.items()},
            connections=self.connections,
            seeded_inputs=seeded,
        )

    # ------------------------------------------------------------------ #
    # Width helpers
    # ------------------------------------------------------------------ #
    def _port_width(self, ref: PortRef) -> int:
        block, port = ref
        return self._blocks_spec[block].in_ports[port]

    def _port_width_out(self, ref: PortRef, blocks) -> int:
        block, port = ref
        return blocks[block].out_ports[port]

    # ------------------------------------------------------------------ #
    # Core engine
    # ------------------------------------------------------------------ #
    def forward(self, u: Tensor, x: Tensor, d: Optional[Tensor] = None):
        batch = x.shape[0]
        signals: Dict[PortRef, Tensor] = {}

        # 1) seed external inputs, disturbances and state taps
        self._seed(signals, self.input_refs, u)
        if d is not None and self.disturbance_refs:
            self._seed(signals, self.disturbance_refs, d)
        for src_block, dst in self.state_taps:
            signals[dst] = x[:, self._state_slices[src_block]]

        # 2) outputs of strictly-proper (non-feedthrough) blocks become sources
        dx_parts: Dict[str, Tensor] = {}
        for name, block in self.blocks.items():
            if block.feedthrough:
                continue
            if block.is_dynamic:
                xb = x[:, self._state_slices[name]]
                outs = block.eval_output(xb, self._block_d(signals, name))
            else:  # static source (e.g. Constant)
                outs = block.evaluate(self._gather(signals, name), batch_size=batch)
            for p, t in outs.items():
                signals[(name, p)] = t

        # 3) topological evaluation of feedthrough blocks
        for name in self.graph.order:
            block = self.blocks[name]
            inputs = self._gather(signals, name)
            if block.is_dynamic:
                xb = x[:, self._state_slices[name]]
                dxb, outs = block.dynamics(inputs, xb)
                dx_parts[name] = dxb
            else:
                outs = block.evaluate(inputs, batch_size=batch)
            for p, t in outs.items():
                signals[(name, p)] = t

        # 4) state derivatives of the strictly-proper dynamic blocks
        for name in self._dyn_names:
            if name in dx_parts:
                continue
            block = self.blocks[name]
            inputs = self._gather(signals, name)
            xb = x[:, self._state_slices[name]]
            dxb, _ = block.dynamics(inputs, xb)
            dx_parts[name] = dxb

        # 5) assemble global dx (dynamic-block order) and external y
        dx = torch.cat([dx_parts[name] for name in self._dyn_names], dim=-1) \
            if self._dyn_names else x.new_zeros((batch, 0))
        y = torch.cat([signals[ref] for ref in self.output_refs], dim=-1) \
            if self.output_refs else u.new_zeros((batch, 0))
        return dx, y

    # ------------------------------------------------------------------ #
    def _seed(self, signals, refs: List[PortRef], vec: Tensor) -> None:
        off = 0
        for ref in refs:
            w = self._port_width(ref)
            signals[ref] = vec[:, off:off + w]
            off += w

    def _gather(self, signals, block_name: str) -> Dict[str, Tensor]:
        """Collect the resolved input tensors for ``block_name`` by port name."""
        inputs: Dict[str, Tensor] = {}
        for port in self.blocks[block_name].in_ports:
            ref = (block_name, port)
            if ref in signals:
                inputs[port] = signals[ref]
            else:
                src = self.graph.incoming[ref]
                inputs[port] = signals[src]
        return inputs

    def _block_d(self, signals, block_name: str) -> Optional[Tensor]:
        ref = (block_name, "d")
        if ref in signals:
            return signals[ref]
        if ref in self.graph.incoming:
            return signals[self.graph.incoming[ref]]
        return None

    # ------------------------------------------------------------------ #
    # SSModel lifecycle
    # ------------------------------------------------------------------ #
    def _frame(self) -> Tuple[Tensor, ...]:
        frames: List[Tensor] = []
        for b in self.blocks.values():
            frames.extend(b._frame())
        return tuple(frames)

    def _right_inverse(self, *args, **kwargs):
        return None

    def init_weights_(self, *args, **kwargs):
        for b in self.blocks.values():
            if hasattr(b, "init_weights_"):
                try:
                    b.init_weights_()
                except (NotImplementedError, TypeError):
                    pass

    def get_block(self, name: str) -> Block:
        return self.blocks[name]

    def trainable_blocks(self, *names: str) -> "Diagram":
        """Freeze every block, then unfreeze only the named ones. Returns self."""
        for b in self.blocks.values():
            b.freeze()
        for n in names:
            self.blocks[n].unfreeze()
        return self

    def clone(self) -> "Diagram":
        cloned_blocks = {name: b.clone() for name, b in self.blocks.items()}
        return Diagram(
            blocks=cloned_blocks,
            connections=self._connections_spec,
            inputs=self._inputs_spec,
            outputs=self._outputs_spec,
            disturbances=self._disturbances_spec,
            state_taps=self._state_taps_spec,
            feedthrough=self.feedthrough,
        )

    def __repr__(self) -> str:
        names = ", ".join(self.blocks.keys())
        return (f"Diagram(blocks=[{names}], nu={self.nu}, ny={self.ny}, "
                f"nx={self.nx}, nd={self.nd})")

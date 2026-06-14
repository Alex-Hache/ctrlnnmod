"""Pure wiring logic for diagrams: connection bookkeeping, evaluation ordering and
algebraic-loop detection. This module is deliberately free of any torch state - it
only manipulates block/port names so it can be unit-tested in isolation.

A *port reference* is the pair ``(block_name, port_name)``. A *connection* wires a
source output port to a destination input port::

    (src_block, src_port) -> (dst_block, dst_port)

The engine must evaluate blocks in an order where every input is available before
the block that consumes it. Only **feedthrough** outputs (those that depend directly
on the block's inputs) create ordering constraints; a strictly-proper dynamic block
(``feedthrough=False``) produces its outputs from its state alone, so it acts as a
*source* and breaks feedback loops. A cycle among feedthrough outputs is an
**algebraic loop** and is reported via :class:`AlgebraicLoopError`.
"""

from typing import Dict, List, Tuple

PortRef = Tuple[str, str]
Connection = Tuple[PortRef, PortRef]  # (source out-port, destination in-port)


class AlgebraicLoopError(RuntimeError):
    """Raised when the feedthrough subgraph contains a cycle (an algebraic loop)."""


class WiringError(ValueError):
    """Raised for structural wiring problems (unknown ports, width mismatch, ...)."""


class WiringGraph:
    """Validated description of how blocks are connected inside a diagram.

    Args:
        block_ports: ``{block_name: (in_ports, out_ports)}`` where each ``*_ports`` is
            a ``{port_name: width}`` mapping.
        block_feedthrough: ``{block_name: bool}`` - whether the block's outputs depend
            directly on its inputs.
        connections: list of ``((src, sport), (dst, dport))`` edges.
        seeded_inputs: set of destination ``(block, port)`` refs that are supplied
            externally (diagram inputs, disturbances) or via state taps, and therefore
            need no incoming connection.
    """

    def __init__(
        self,
        block_ports: Dict[str, Tuple[Dict[str, int], Dict[str, int]]],
        block_feedthrough: Dict[str, bool],
        connections: List[Connection],
        seeded_inputs: List[PortRef],
    ) -> None:
        self.block_ports = block_ports
        self.block_feedthrough = block_feedthrough
        self.connections = list(connections)
        self.seeded_inputs = set(seeded_inputs)
        # destination port -> source port
        self.incoming: Dict[PortRef, PortRef] = {}
        self._validate()
        self.order = self._topological_order()

    # ------------------------------------------------------------------ #
    def _validate(self) -> None:
        for (src, sport), (dst, dport) in self.connections:
            if src not in self.block_ports:
                raise WiringError(f"Unknown source block '{src}'")
            if dst not in self.block_ports:
                raise WiringError(f"Unknown destination block '{dst}'")
            if sport not in self.block_ports[src][1]:
                raise WiringError(f"Block '{src}' has no output port '{sport}'")
            if dport not in self.block_ports[dst][0]:
                raise WiringError(f"Block '{dst}' has no input port '{dport}'")
            w_src = self.block_ports[src][1][sport]
            w_dst = self.block_ports[dst][0][dport]
            if w_src != w_dst:
                raise WiringError(
                    f"Port width mismatch on {src}.{sport}->{dst}.{dport}: "
                    f"{w_src} != {w_dst}"
                )
            if (dst, dport) in self.incoming:
                raise WiringError(f"Input {dst}.{dport} is driven by two sources")
            self.incoming[(dst, dport)] = (src, sport)

        # every input port must be driven (connection) or seeded (external/tap)
        for block, (in_ports, _) in self.block_ports.items():
            for port in in_ports:
                ref = (block, port)
                if ref not in self.incoming and ref not in self.seeded_inputs:
                    raise WiringError(
                        f"Input {block}.{port} is unconnected (no source and not seeded)"
                    )

    # ------------------------------------------------------------------ #
    def _topological_order(self) -> List[str]:
        """Return an evaluation order for the *feedthrough* blocks.

        Blocks whose outputs are sources (strictly-proper dynamic blocks) are not
        ordered here - their outputs are available before the pass begins. A cycle
        among feedthrough blocks raises :class:`AlgebraicLoopError`.
        """
        feedthrough_blocks = [b for b, ft in self.block_feedthrough.items() if ft]
        ft_set = set(feedthrough_blocks)

        # Build edges src->dst when src is a feedthrough block feeding dst's input.
        deps: Dict[str, set] = {b: set() for b in feedthrough_blocks}
        for (dst, _dport), (src, _sport) in self.incoming.items():
            if dst in ft_set and src in ft_set:
                deps[dst].add(src)

        order: List[str] = []
        visited: Dict[str, int] = {}  # 0=visiting, 1=done

        def visit(node: str, stack: List[str]) -> None:
            state = visited.get(node)
            if state == 1:
                return
            if state == 0:
                cycle = stack[stack.index(node):] + [node]
                raise AlgebraicLoopError(
                    "Algebraic loop detected among feedthrough blocks: "
                    + " -> ".join(cycle)
                    + ". Break it by making a block in the loop strictly proper "
                    "(feedthrough=False) or by inserting a dynamic block."
                )
            visited[node] = 0
            for dep in sorted(deps[node]):
                visit(dep, stack + [node])
            visited[node] = 1
            order.append(node)

        for b in sorted(feedthrough_blocks):
            visit(b, [])
        return order

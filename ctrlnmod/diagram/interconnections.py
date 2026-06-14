"""Ready-made interconnection patterns built on top of :class:`Diagram`.

These are thin convenience constructors that emit a fully-wired :class:`Diagram`
for the most common structures, so users rarely need to spell out connections by
hand. Everything they return is an ordinary :class:`Diagram` (hence an
:class:`~ctrlnmod.models.ssmodels.base.SSModel`) and simulates/trains as usual.
"""

from typing import List, Optional

from ctrlnmod.blocks.base import Block
from ctrlnmod.blocks.primitives import Gain
from ctrlnmod.controllers.base import Controller
from ctrlnmod.models.ssmodels.base import SSModel
from ctrlnmod.diagram.diagram import Diagram


def Series(blocks: List[SSModel], names: Optional[List[str]] = None) -> Diagram:
    """Cascade ``y`` of each block into ``u`` of the next: ``b0 -> b1 -> ... -> bn``.

    The external input feeds the first block's ``u`` and the external output is the
    last block's ``y``. Assumes single ``u``/``y`` ports of matching widths.
    """
    if names is None:
        names = [f"b{i}" for i in range(len(blocks))]
    block_map = dict(zip(names, blocks))
    connections = [
        (f"{names[i]}.y", f"{names[i + 1]}.u") for i in range(len(blocks) - 1)
    ]
    return Diagram(
        blocks=block_map,
        connections=connections,
        inputs=[f"{names[0]}.u"],
        outputs=[f"{names[-1]}.y"],
    )


def Parallel(blocks: List[SSModel], names: Optional[List[str]] = None) -> Diagram:
    """Drive every block with the **same** external input; concatenate their outputs.

    A passthrough (identity :class:`~ctrlnmod.blocks.primitives.Gain`) fans the
    external input out to each block, so all blocks must share the same input width.
    """
    if names is None:
        names = [f"b{i}" for i in range(len(blocks))]
    nu = blocks[0].nu
    block_map = {"split": Gain(nu)}
    block_map.update(dict(zip(names, blocks)))
    connections = [("split.out", f"{n}.u") for n in names]
    return Diagram(
        blocks=block_map,
        connections=connections,
        inputs=["split.in"],
        outputs=[f"{n}.y" for n in names],
    )


def FeedbackInterconnection(
    plant: SSModel,
    controller: Controller,
    plant_name: str = "plant",
    controller_name: str = "controller",
) -> Diagram:
    r"""Wire a controller in feedback around a plant.

    The controller consumes the external reference (its ``u`` port), the plant
    measurement and optional disturbance, and drives the plant input::

        ref --> controller.u
        plant.y --> controller.y      (output feedback)   OR
        plant.state --> controller.x  (state feedback)
        controller.v --> plant.u
        y := plant.y

    Whether output- or state-feedback is used is inferred from the controller's
    input ports (``y`` vs ``x``). This reproduces the structure of
    :class:`~ctrlnmod.models.ssmodels.continuous.linearization.feedbacklin2.FLNSSM`
    using the generic engine.
    """
    blocks = {plant_name: plant, controller_name: controller}
    connections = [(f"{controller_name}.v", f"{plant_name}.u")]
    state_taps = None
    if "y" in controller.in_ports:
        connections.append((f"{plant_name}.y", f"{controller_name}.y"))
    elif "x" in controller.in_ports:
        state_taps = [(plant_name, f"{controller_name}.x")]
    else:
        raise ValueError("Controller must expose a 'y' (output) or 'x' (state) input port")

    disturbances = None
    dist_refs = []
    if plant.nd is not None:
        dist_refs.append(f"{plant_name}.d")
    if "d" in controller.in_ports:
        dist_refs.append(f"{controller_name}.d")
    if dist_refs:
        disturbances = dist_refs

    return Diagram(
        blocks=blocks,
        connections=connections,
        inputs=[f"{controller_name}.u"],
        outputs=[f"{plant_name}.y"],
        disturbances=disturbances,
        state_taps=state_taps,
    )

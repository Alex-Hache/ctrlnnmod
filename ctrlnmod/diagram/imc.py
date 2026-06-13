r"""Internal Model Control (IMC) as a differentiable diagram.

The classic IMC structure places a learnable controller :math:`Q` and an internal
model :math:`M` alongside the plant :math:`P`::

        r ->(+)-->[ Q ]--+--> u -->[ P ]----+----> y
             ^- dhat     |                  |
                         +------>[ M ]--ym---|--(+/-)--> dhat = y - ym

Concretely, with the disturbance estimate ``dhat = y - ym`` and modified error
``e = r - dhat``:

.. math::
    u = Q(e), \qquad y = P(u), \qquad y_m = M(u), \qquad d̂ = y - y_m .

Because :math:`P` and :math:`M` are strictly proper (``feedthrough=False``), their
outputs are available before ``u`` is known, so the loop has **no algebraic loop**.

Everything is an ordinary :class:`~ctrlnmod.diagram.diagram.Diagram`, so it
simulates with :class:`~ctrlnmod.integrators.RK4Simulator` and trains with
:class:`~ctrlnmod.train.LitNode`. Use :meth:`Diagram.trainable_blocks` to choose
what to learn (``"model"``, ``"controller"``, or both); a known/identified plant is
typically frozen.
"""

from typing import List, Optional, Union

from torch import Tensor

from ctrlnmod.blocks.primitives import Sum
from ctrlnmod.controllers.base import Controller
from ctrlnmod.controllers.neural import NeuralController
from ctrlnmod.models.ssmodels.base import SSModel
from ctrlnmod.diagram.diagram import Diagram


class IMC(Diagram):
    """Internal Model Control loop.

    Args:
        plant: The plant :math:`P` (an :class:`SSModel`). Often fixed/identified and
            frozen during controller/model learning.
        model: The internal model :math:`M` (an :class:`SSModel`) approximating the
            plant. Same ``nu``/``ny`` as the plant.
        controller: Optional controller :math:`Q` mapping the error port ``e`` (width
            ``ny``) to a control port ``u`` (width ``nu``). If ``None``, a
            :class:`~ctrlnmod.controllers.neural.NeuralController` is created.
        hidden_layers: Hidden widths for the default controller.
        act_f: Activation for the default controller.
        lip: Optional Lipschitz bound for the default controller.
    """

    def __init__(
        self,
        plant: SSModel,
        model: SSModel,
        controller: Optional[Controller] = None,
        hidden_layers: List[int] = (32, 32),
        act_f: str = "tanh",
        lip: Optional[Union[float, Tensor]] = None,
    ) -> None:
        if plant.nu != model.nu or plant.ny != model.ny:
            raise ValueError(
                f"Plant and internal model must share nu/ny, got "
                f"P(nu={plant.nu}, ny={plant.ny}) vs M(nu={model.nu}, ny={model.ny})"
            )
        nu, ny = plant.nu, plant.ny
        if controller is None:
            controller = NeuralController(
                n_in=ny, n_out=nu, hidden_layers=list(hidden_layers),
                act_f=act_f, in_port="e", out_port="u", lip=lip,
            )
        # keep construction args as attributes so get_config/clone work
        self._imc_args = dict(hidden_layers=list(hidden_layers), act_f=act_f, lip=lip)

        blocks = {
            "plant": plant,
            "model": model,
            "controller": controller,
            "sum_d": Sum(ny, signs=["+", "-"]),   # dhat = y - ym
            "sum_e": Sum(ny, signs=["+", "-"]),   # e    = r - dhat
        }
        connections = [
            ("plant.y", "sum_d.in0"),
            ("model.y", "sum_d.in1"),
            ("sum_d.out", "sum_e.in1"),
            ("sum_e.out", "controller.e"),
            ("controller.u", "plant.u"),
            ("controller.u", "model.u"),
        ]
        super().__init__(
            blocks=blocks,
            connections=connections,
            inputs=["sum_e.in0"],   # external reference r
            outputs=["plant.y"],    # external output y
        )

    # convenient accessors -------------------------------------------------
    @property
    def plant(self) -> SSModel:
        return self.blocks["plant"]

    @property
    def model(self) -> SSModel:
        return self.blocks["model"]

    @property
    def controller(self) -> Controller:
        return self.blocks["controller"]

    def clone(self) -> "IMC":
        return IMC(
            plant=self.blocks["plant"].clone(),
            model=self.blocks["model"].clone(),
            controller=self.blocks["controller"].clone(),
            **self._imc_args,
        )

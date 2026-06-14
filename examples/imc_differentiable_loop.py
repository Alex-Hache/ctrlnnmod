"""Internal Model Control (IMC) as a differentiable loop.

This example shows how to assemble a closed loop from reusable blocks and train a
neural controller (and, optionally, the internal model) end-to-end through the
simulator - the "Simulink for neural control" workflow.

Pipeline
--------
1. Build a (fixed) plant ``P`` and an internal model ``M`` as ``SSModel`` blocks.
2. Wire them with a neural controller ``Q`` into an :class:`IMC` diagram. Because a
   diagram *is* an ``SSModel``, it plugs straight into ``RK4Simulator``.
3. Choose what to train via :meth:`Diagram.trainable_blocks` (here: the controller).
4. Optimise a setpoint-tracking loss; gradients flow through the whole rollout.

Run with::

    python examples/imc_differentiable_loop.py
"""

import torch

from ctrlnmod.models.ssmodels.continuous.linear import SSLinear
from ctrlnmod.controllers import NeuralController
from ctrlnmod.diagram import IMC
from ctrlnmod.integrators import RK4Simulator
from ctrlnmod.losses import ReferenceTrackingLoss


def stable_plant(seed: int) -> SSLinear:
    """A small stable, controllable SISO plant with hand-set matrices."""
    torch.manual_seed(seed)
    p = SSLinear(input_dim=1, output_dim=1, state_dim=2)
    with torch.no_grad():
        p.A.weight.copy_(torch.tensor([[-1.0, 0.6], [0.0, -2.0]]))
        p.B.weight.copy_(torch.tensor([[1.0], [1.0]]))
        p.C.weight.copy_(torch.tensor([[1.0, 0.0]]))
    return p


def main() -> None:
    torch.manual_seed(0)

    # 1) plant and internal model (slightly mismatched, as in practice)
    plant = stable_plant(seed=1)
    model = stable_plant(seed=2)

    # 2) controller Q: maps the IMC error e (dim ny) to a control u (dim nu)
    controller = NeuralController(n_in=plant.ny, n_out=plant.nu,
                                  hidden_layers=[32, 32], act_f="tanh")

    # 3) assemble the IMC loop; it behaves as a single SSModel
    imc = IMC(plant, model, controller=controller)
    print(imc)

    # learn the controller only; freeze the (known) plant and the internal model
    imc.trainable_blocks("controller")

    sim = RK4Simulator(imc, ts=2e-2)
    criterion = ReferenceTrackingLoss()
    opt = torch.optim.Adam([p for p in imc.parameters() if p.requires_grad], lr=5e-3)

    # 4) setpoint-tracking task: follow a unit step reference
    batch, horizon = 8, 50
    r = torch.ones(batch, horizon, 1)
    x0 = torch.zeros(batch, imc.nx)

    for epoch in range(60):
        opt.zero_grad()
        _, y = sim(r, x0)
        loss = criterion(y, r)
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            print(f"epoch {epoch:3d} | tracking loss {loss.item():.5f}")

    with torch.no_grad():
        _, y = sim(r, x0)
        steady = y[:, -1, 0].mean().item()
    print(f"final tracking loss {criterion(y, r).item():.5f} | "
          f"mean steady-state output {steady:.3f} (target 1.0)")


if __name__ == "__main__":
    main()

from torch.nn import Module
from ctrlnmod.models.feedforward.lbdn import Fxu, LipFxu, LipHx, Hx
from ctrlnmod.lmis.hinf import HInfCont
from torch.nn.init import zeros_
from torch.linalg import eigvals
from torch import Tensor, real, min
from typing import Tuple
from ctrlnmod.models.ssmodels.linear import NnLinear
from ctrlnmod.models.ssmodels.hinf import L2BoundedLinear
import torch
import torch.nn as nn
from math import sqrt
import numpy as np
from scipy.io import savemat
from geotorch_custom import is_parametrized





def rk4_discretize(A, h):
    I = torch.eye(A.size(0), dtype=A.dtype, device=A.device)
    k1 = h * A
    k2 = h * torch.mm(A, (I + 0.5 * k1))
    k3 = h * torch.mm(A, (I + 0.5 * k2))
    k4 = h * torch.mm(A, (I + k3))
    return I + (k1 + 2*k2 + 2*k3 + k4) / 6.0


class Grnssm(Module):
    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nh: int,
        n_hidden_layers: int = 1,
        actF: str = 'tanh',
        out_eq_nl: bool = False,
        alpha: float = None,
        bias: bool = True
    ) -> None:
        super(Grnssm, self).__init__()

        self.nu = nu
        self.nx = nx
        self.ny = ny
        self.nh = nh
        self.n_hidden_layers = n_hidden_layers
        self.act_name = actF
        self.bias = bias
        self.out_eq_nl = out_eq_nl
        self.alpha = alpha

        if actF.lower() == 'tanh':
            self.actF = nn.Tanh()
        elif actF.lower() == 'relu':
            self.actF = nn.ReLU()
        else:
            raise NotImplementedError(f"Function {actF} not yet implemented. Please choose 'tanh' or 'relu'.")

        self.linmod = NnLinear(self.nu, self.ny, self.nx, alpha=alpha)
        self.fx = Fxu(self.nu, self.nh, self.nx, actF=self.actF, n_hidden=n_hidden_layers, bias=self.bias)
        
        if self.out_eq_nl:
            self.hx = Hx(self.nx, self.nh, self.ny, actF=self.actF, n_hidden=n_hidden_layers, bias=bias)

    def __repr__(self):
        return f"GRNSSM : nu={self.nu} nx={self.nx} nh={self.nh} ny={self.ny} activation={self.act_name}"

    def __str__(self) -> str:
        return "GRNSSM"

    def forward(self, u, x):
        x_lin, y_lin = self.linmod(u, x)
        fx = self.fx(x, u)
        dx = x_lin + fx

        if self.out_eq_nl:
            hx = self.hx(x)
            y = y_lin + hx
        else:
            y = y_lin
        return dx, y

    def init_weights_(self, A0, B0, C0, requires_grad=True, adjust_alpha=False, margin=1e-4) -> None:
        self.linmod.init_weights_(A0, B0, C0, requires_grad=requires_grad, adjust_alpha=adjust_alpha, margin=margin)
        zeros_(self.fx.Wout.weight)
        if self.fx.Wout.bias is not None:
            zeros_(self.fx.Wout.bias)
        if self.out_eq_nl:
            zeros_(self.hx.Wout.weight)
            if self.hx.Wout.bias is not None:
                zeros_(self.hx.Wout.bias)

        nn.init.xavier_uniform_(self.fx.Wfu, gain=nn.init.calculate_gain(self.act_name))
        nn.init.xavier_uniform_(self.fx.Wfx, gain=nn.init.calculate_gain(self.act_name))

    def clone(self):
        copy = type(self)(self.nu, self.ny, self.nx, self.nh, self.n_hidden_layers, self.act_name, self.out_eq_nl, self.alpha, self.bias)
        copy.load_state_dict(self.state_dict())
        return copy

    def check(self, *args):
        return True, {}

    def _frame(self):
        A = self.linmod.A.weight
        B2 = self.linmod.B.weight
        C2 = self.linmod.C.weight

        list_shapes_x, nq_x = self._build_dims(self.fx)
        if self.out_eq_nl:
            list_shapes_y, nq_y = self._build_dims(self.hx)
        else:
            list_shapes_y, nq_y = [], 0

        B1 = self._build_B1(nq_x, list_shapes_x, nq_y)
        C1, D12 = self._build_C1_D12(nq_x, list_shapes_x, nq_y, list_shapes_y)
        D11 = self._build_D11(nq_x, list_shapes_x, nq_y, list_shapes_y)
        D21 = self._build_D21(nq_y, list_shapes_y, nq_x)
        D22 = torch.zeros(self.ny, self.nu)

        return A, B1, B2, C1, C2, D11, D12, D21, D22

    def _build_dims(self, nl_part):
        list_shapes = []
        for layer in nl_part.layers:
            if hasattr(layer, 'weight'):
                list_shapes.append(layer.weight.shape)
        nq = sum(shape[0] for shape in list_shapes[:-1])
        return list_shapes, nq

    def _build_C1_D12(self, nq_x, list_shapes_x, nq_y, list_shapes_y):
        C1_x = torch.zeros((nq_x, self.nx))
        D12_x = torch.zeros((nq_x, self.nu))
        nh1, nxu = list_shapes_x[0]
        C1_x[:nh1, :] = self.fx.layers[0].weight[:, :self.nx]
        D12_x[:nh1, :] = self.fx.layers[0].weight[:, self.nx:]

        if self.out_eq_nl:
            C1_y = torch.zeros((nq_y, self.nx))
            D12_y = torch.zeros((nq_y, self.nu))
            nh1y, nyu = list_shapes_y[0]
            C1_y[:nh1y, :] = self.hx.layers[0].weight[:, :self.nx]
            C1 = torch.cat((C1_x, C1_y), dim=0)
            D12 = torch.cat((D12_x, D12_y), dim=0)
        else:
            C1 = C1_x
            D12 = D12_x
        return C1, D12

    def _build_B1(self, nq_x, list_shapes_x, nq_y):
        B1_x = torch.zeros((self.nx, nq_x))
        nl, nhl = list_shapes_x[-1]
        B1_x[:, nq_x - nhl:] = self.fx.layers[-1].weight

        B1_y = torch.zeros((self.nx, nq_y))
        B1 = torch.cat([B1_x, B1_y], dim=1)
        return B1

    def _build_D21(self, nq_y, list_shapes_y, nq_x):
        D21 = torch.zeros((self.ny, nq_x + nq_y))
        if self.out_eq_nl:
            nl, nhl = list_shapes_y[-1]
            D21[:, (nq_x + nq_y) - nhl:] = self.hx.layers[-1].weight
        return D21

    def _build_D11(self, nq_x, list_shapes_x, nq_y, list_shapes_y):
        list_shapes_inter = list_shapes_x[1:-1]
        D11_x = torch.zeros((nq_x, nq_x))
        index_row, index_col = list_shapes_x[0][0], 0
        for i, shape in enumerate(list_shapes_inter):
            end_row = index_row + shape[0]
            end_col = index_col + shape[1]
            D11_x[index_row:end_row, index_col:end_col] = self.fx.layers[2 * (i + 1)].weight
            index_row = end_row
            index_col = end_col

        D11_y = torch.zeros((nq_y, nq_y))
        if self.out_eq_nl:
            index_row, index_col = list_shapes_y[0][0], 0
            for i, shape in enumerate(list_shapes_inter):
                end_row = index_row + shape[0]
                end_col = index_col + shape[1]
                D11_y[index_row:end_row, index_col:end_col] = self.hx.layers[2 * (i + 1)].weight
                index_row = end_row
                index_col = end_col
        D11 = torch.block_diag(D11_x, D11_y)
        return D11

class LipGrnssm(Grnssm):
    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nh: int,
        n_hidden_layers: int = 1,
        actF: str = 'tanh',
        out_eq_nl: bool = False,
        lip: Tuple[Tensor, Tensor] = (Tensor([1]), Tensor([1])),
        alpha: float = None,
        param: str = 'expm',
        bias: bool = True
    ) -> None:
        super(LipGrnssm, self).__init__(nu, ny, nx, nh, n_hidden_layers, actF, out_eq_nl, alpha, bias)

        self.lip_x = lip[0]
        self.lip_u = lip[1]

        self.fx = LipFxu(self.nu, self.nh, self.nx, n_hidden=n_hidden_layers, scalex=self.lip_x, scaleu=self.lip_u, param=param, bias=bias)
        if self.out_eq_nl:
            self.hx = LipHx(self.nx, self.nh, self.ny, n_hidden=n_hidden_layers, scalex=self.lip_x, bias=bias, param=param)

    def __str__(self):
        return 'lipgnssm'
    
    def __repr__(self):
        return f"LipGNSSM  : alpha = {self.alpha}   Lamda x = {self.lip_x}, Lambda_u = {self.lip_u}"
        
    def check(self):
        fx_check, infos = self.fx.check()
        merged_info = infos.copy()  # Start with the infos dictionary
        
        # Initialize the overall check as the result of fx_check
        overall_check = fx_check
        
        if self.out_eq_nl:
            hx_check, infos_hx = self.hx.check()
            overall_check = overall_check and hx_check  # Combine the boolean checks
            merged_info.update(infos_hx)  # Merge the dictionaries
        
        return overall_check, merged_info


    def _frame(self):
        A = self.linmod.A.weight
        B2 = self.linmod.B.weight
        C2 = self.linmod.C.weight
        weights_x = self.fx.extractWeightsSandwich()
        list_shapes_x, nq_x = self._build_dims(weights_x)
        if self.out_eq_nl:
            weights_y = self.hx.extractWeightsSandwich()
            list_shapes_y, nq_y = self._build_dims(weights_y)
        else:
            list_shapes_y, nq_y, weights_y = [], 0, []

        B1 = self._build_B1(nq_x, list_shapes_x, nq_y, weights_x)
        C1, D12 = self._build_C1_D12(nq_x, list_shapes_x, nq_y, list_shapes_y,weights_x, weights_y)
        D11 = self._build_D11(nq_x, list_shapes_x, nq_y, list_shapes_y, weights_x, weights_y)
        D21 = self._build_D21(nq_y, list_shapes_y, nq_x, weights_y)
        D22 = torch.zeros(self.ny, self.nu)

        return A, B1, B2, C1, C2, D11, D12, D21, D22

    def _build_dims(self, weights):
        list_shapes = []
        for layer in weights:
            list_shapes.append(layer.shape)
        nq = sum(shape[0] for shape in list_shapes[:-1])
        return list_shapes, nq

    def _build_C1_D12(self, nq_x, list_shapes_x, nq_y, list_shapes_y, weights_x, weights_y):
        C1_x = torch.zeros((nq_x, self.nx))
        D12_x = torch.zeros((nq_x, self.nu))
        nh1, nxu = list_shapes_x[0]
        C1_x[:nh1, :] = weights_x[0][:, :self.nx]
        D12_x[:nh1, :] =  weights_x[0][:, self.nx:]

        if self.out_eq_nl:
            C1_y = torch.zeros((nq_y, self.nx))
            D12_y = torch.zeros((nq_y, self.nu))
            nh1y, nyu = list_shapes_y[0]
            C1_y[:nh1y, :] =  weights_y[0][:, :self.nx]
            C1 = torch.cat((C1_x, C1_y), dim=0)
            D12 = torch.cat((D12_x, D12_y), dim=0)
        else:
            C1 = C1_x
            D12 = D12_x
        return C1, D12

    def _build_B1(self, nq_x, list_shapes_x, nq_y, weights_x):
        B1_x = torch.zeros((self.nx, nq_x))
        nl, nhl = list_shapes_x[-1]
        B1_x[:, nq_x - nhl:] = weights_x[-1]

        B1_y = torch.zeros((self.nx, nq_y))
        B1 = torch.cat([B1_x, B1_y], dim=1)
        return B1

    def _build_D21(self, nq_y, list_shapes_y, nq_x, weights_y):
        D21 = torch.zeros((self.ny, nq_x + nq_y))
        if self.out_eq_nl:
            nl, nhl = list_shapes_y[-1]
            D21[:, (nq_x + nq_y) - nhl:] = weights_y[-1]
        return D21

    def _build_D11(self, nq_x, list_shapes_x, nq_y, list_shapes_y, weights_x, weights_y):
        list_shapes_inter = list_shapes_x[1:-1]
        D11_x = torch.zeros((nq_x, nq_x))
        index_row, index_col = list_shapes_x[0][0], 0
        for i, shape in enumerate(list_shapes_inter):
            end_row = index_row + shape[0]
            end_col = index_col + shape[1]
            D11_x[index_row:end_row, index_col:end_col] = weights_x[2 *i + 1]
            index_row = end_row
            index_col = end_col

        D11_y = torch.zeros((nq_y, nq_y))
        if self.out_eq_nl:
            index_row, index_col = list_shapes_y[0][0], 0
            for i, shape in enumerate(list_shapes_inter):
                end_row = index_row + shape[0]
                end_col = index_col + shape[1]
                D11_y[index_row:end_row, index_col:end_col] = weights_y[2 *i + 1]
                index_row = end_row
                index_col = end_col
        D11 = torch.block_diag(D11_x, D11_y)
        return D11


class StableGNSSM(LipGrnssm):
    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nh: int,
        n_hidden_layers: int = 1,
        actF: str = 'tanh',
        out_eq_nl: bool = False,
        lmbd: float = 1.0,
        epsilon: float = 1e-2,
        bias: bool = True
    ):
        super(StableGNSSM, self).__init__(
            nu, ny, nx, nh, n_hidden_layers, actF, out_eq_nl,
            lip=(Tensor([lmbd]), Tensor([lmbd])), alpha=lmbd + epsilon, bias=bias
        )

class L2IncGrNSSM(LipGrnssm):
    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        nh: int,
        n_hidden_layers: int = 1,
        actF: str = 'tanh',
        out_eq_nl: bool = False,
        l2i: float = 1.0,
        alpha: float = 1.0,
        bias: bool = True
    ) -> None:
        self.gamma = l2i
        lipx = torch.sqrt(torch.Tensor([2 * alpha]))
        lip = (lipx, l2i / torch.sqrt(torch.Tensor([2])))
        super().__init__(nu, ny, nx, nh, n_hidden_layers, actF, out_eq_nl, lip, alpha, bias=bias)
        scaleH = 1 / sqrt(2) - 0.1
        self.linmod = L2BoundedLinear(nu, ny, nx, gamma=l2i, alpha=alpha, scaleH=scaleH, epsilon=2.0)
        self._frame()

    def __repr__(self):
        return f"Incr L2 bounded GRNSSM : eta={float(self.gamma)} -- nh={self.nh}"

    def _frame(self):
        self.alpha = (self.lip_x**2) / (2 * min(real(eigvals(self.linmod.P))))
        self.linmod.alpha = self.alpha
        return super()._frame()

    def right_inverse_(self, A, B, C, eta, alpha):
        self.linmod.right_inverse_(A, B, C, eta, alpha)

    def check_(self, traj=None, N_trys=1000) -> Tuple[bool, np.ndarray]:
        self._frame()
        gammas = []
        if traj is not None:
            u_traj, x_traj = traj
            for u_eq, x_eq in zip(u_traj, x_traj):
                gamma = self.compute_L2_taylor(u_eq.unsqueeze(0), x_eq.unsqueeze(0))
                gammas.append(gamma)
        else:
            for _ in range(N_trys):
                u_eq, x_eq = torch.rand((1, self.nu)), torch.rand((1, self.nx))
                gamma = self.compute_L2_taylor(u_eq, x_eq)
                gammas.append(gamma)
        bl2i = all(np.array(gammas) <= self.gamma)
        b_lin_bounded, _ = self.linmod.check_()
        if not b_lin_bounded:
            print("Prescribed L2 gain for linear part not okay")
        bLipschitz = self.fx.check_()
        if not bLipschitz:
            print("Prescribed Lipschitz constant for nonlinear part not okay")

        return (bl2i and b_lin_bounded and bLipschitz), np.array(gammas)

    def compute_L2_taylor(self, u_eq, x_eq, epsilon=1e-7, solver="MOSEK"):
        inputs = (u_eq, x_eq)
        jac = torch.autograd.functional.jacobian(self.forward, inputs)
        B, A = jac[0]
        D, C = jac[1]
        with torch.no_grad():
            _, gamma, _ = HInfCont.solve(A, B, C, D, alpha = torch.zeros(1))
        return gamma

    def save_weights(self):
        A, B, C = self.linmod.eval_()

        weights = self.fx.extractWeightsSandwich()
        Wfx = weights[0][:, :self.nx]
        Wfu = weights[0][:, self.nx:]
        Wf = weights[1]

        mdict = {
            'Wfx': Wfx.detach().numpy(),
            'Wfu': Wfu.detach().numpy(),
            'Wf': Wf.detach().numpy(),
            'A': A.detach().numpy(),
            'B': B.detach().numpy(),
            'C': C.detach().numpy()
        }
        savemat('weights.mat', mdict)


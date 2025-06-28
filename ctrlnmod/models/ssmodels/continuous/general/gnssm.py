import torch
import torch.nn as nn
from torch.nn import Module
from ctrlnmod.models.feedforward.lbdn import LipFxu, FFNN, LBDN
from ctrlnmod.lmis.hinf import HInfCont
from ctrlnmod.lmis.h2 import H2Cont
from torch.nn.init import zeros_
from torch.linalg import eigvals
from torch import Tensor, real, min
from typing import Tuple
from ..linear import SSLinear, L2BoundedLinear, H2Linear, ExoSSLinear, ExoL2BoundedLinear, ExoH2Linear
from ctrlnmod.utils import parse_act_f
from ctrlnmod.models.ssmodels.base import SSModel
from ctrlnmod.linalg.utils import get_lyap_exp
from typing import Optional, List

"""
    TODO : Implement ExoGNSSM if we need a general model parameterized with exogenous signals.
"""
class GNSSM(SSModel):
    r"""
    Class representing a Generalized Neural State-Space Model (GNSSM).
    ..math::
        \dot{x} = Ax + Bu + f(x, u)
        y = Cx + h(x)
    where :math:`f` and :math:`h` are nonlinear functions approximated by neural networks.
    The linear part :math:`(A, B, C)`can be parameterized to have desired properties such as stability or finite gain.

    Attributes:
        nu (int): Number of inputs.
        ny (int): Number of outputs.
        nx (int): Number of states.
        hidden_layers (List[int]): List of hidden layer sizes for the neural networks.
        act_f (str): Activation function used in the neural networks ('tanh' or 'relu') default is 'relu'.
        out_eq_nl (bool): Whether to include a nonlinear output equation.
        alpha (Optional[float]): Maximum decay rate for the linear part (A) 
            (default is None, which means no decay constraint).
        bias (bool): Whether to include bias terms in the neural networks.
        linmod (Module): a submodule representing the linear part it can be parameterized
    """

    def __init__(
        self,
        nu: int,
        ny: int,
        nx: int,
        hidden_layers: List[int] = [16],
        act_f: str = 'relu',
        out_eq_nl: bool = False,
        alpha: Optional[float] = None,
        bias: bool = True,
        lin_model_type: Optional[str] = None,
        lip: Optional[dict[str, float]] = None,
        param: str = 'sqrtm',
        gamma: Optional[float] = None,
        gamma2: Optional[float] = None
    ) -> None:
        super(GNSSM, self).__init__(nu, ny, nx)
        
        self.hidden_layers = hidden_layers
        self.act_f = parse_act_f(act_f)
        self.act_f_str = act_f
        self.out_eq_nl = out_eq_nl
        self.bias = bias
        self.lin_model_type = lin_model_type

        self.param = param
        self.lip = lip
        self.gamma = gamma
        self.gamma2 = gamma2
        
        if lin_model_type is None:
            self.linmod = SSLinear(self.nu, self.ny, self.nx, alpha=alpha)
        elif lin_model_type.lower() == 'h2':
            if gamma2 is  None:
                raise ValueError("Parameter 'gamma2' must be provided for H2 parameterization.")
            self.gamma2 = gamma2
            self.linmod = H2Linear(self.nu, self.ny, self.nx, gamma2)
        elif lin_model_type.lower() == 'l2':
            if gamma is None:
                raise ValueError("Parameter 'gamma' must be provided for L2 parameterization.")
            if alpha is None:
                alpha = 1e-3
            self.linmod = L2BoundedLinear(self.nu, self.ny, self.nx, gamma, alpha=alpha, param=param)
            self.gamma = gamma

        else:
            raise NotImplementedError(
                f"Parameterization {param} not implemented. Choose 'h2' or 'l2'.")

        self.alpha = alpha

        # Nonlienar part setup
        if lip is not None:
            lip_fx = lip.get('lip_fx', None)
            lip_fu = lip.get('lip_fu', None)
            lip_hx = lip.get('lip_hx', None)
            if lip_fx is None and lip_fu is None and lip_hx is None:
                raise ValueError(f"Invalid dictionnary found {lip} \n At least 1 key must not be equal to None")
            elif lip_fx is not None and lip_fu is None:
                lip_fu = lip_fx
                self.fxu = LBDN(nu+nx, hidden_layers, nx, scale=Tensor([lip_fx]*nx + [lip_fu]*nu), act_f=act_f,bias=bias)
            elif lip_fu is not None and lip_fx is None:
                lip_fx = lip_fu
                self.fxu = LBDN(nu+nx, hidden_layers, nx, scale=Tensor([lip_fx]*nx + [lip_fu]*nu), act_f=act_f,bias=bias)
            elif lip_fx is None and lip_fu is None:
                self.fxu = FFNN(nu+nx, hidden_layers, nx, act_f=act_f, bias=self.bias)
            else:
                self.fxu = LBDN(nu+nx, hidden_layers, nx, scale=Tensor([lip_fx]*nx + [lip_fu]*nu), act_f=act_f,bias=bias)

            if self.out_eq_nl:
                if lip_hx is None:
                    raise ValueError("Please specify a value for output equation nonlinear part found None")
                self.hx = LBDN(nx, hidden_layers, ny, scale=Tensor([lip_hx]*nx), bias=bias)
            else:
                if lip_hx is not None:
                    raise ValueError("lip_hx cannot be set to a non None value and out_eq_nl to False.")
        else:
            self.fxu = FFNN(nx+nu, hidden_layers, nx, act_f, bias)
            if self.out_eq_nl:
                self.hx = FFNN(nx, hidden_layers, ny,act_f, bias)


    def __repr__(self) -> str:
        linmod_type = type(self.linmod).__name__
        repr_str = (
            f"GNSSM(\n"
            f"  nu={self.nu}, ny={self.ny}, nx={self.nx},\n"
            f"  hidden_layers={self.hidden_layers},\n"
            f"  activation='{self.act_f_str}',\n"
            f"  out_eq_nl={self.out_eq_nl}, bias={self.bias},\n"
            f"  alpha={self.alpha},\n"
            f"  linear_param='{linmod_type}',\n"
        )
        
        if hasattr(self, "fxu") and isinstance(self.fxu, LBDN):
            scales = self.fxu.scale.tolist()
            lip_fx = scales[:self.nx]
            lip_fu = scales[self.nx:]
            repr_str += (
                f"  lip_fx={lip_fx[0] if all(x == lip_fx[0] for x in lip_fx) else lip_fx},\n"
                f"  lip_fu={lip_fu[0] if all(x == lip_fu[0] for x in lip_fu) else lip_fu},\n"
            )
        if hasattr(self, "hx") and isinstance(self.hx, LBDN):
            lip_hx = self.hx.scale.tolist()[0]
            repr_str += f"  lip_hx={lip_hx},\n"

        repr_str += ")"
        return repr_str

    def __str__(self) -> str:
        return self.__repr__()
    
    def forward(self, u, x, d=None):
        """
        Forward pass of the GNSSM.
        
        Args:
            u (Tensor): Input tensor.
            x (Tensor): State tensor.
            d (Tensor, optional): Disturbance tensor. Defaults to None.
        
        Returns:
            dx (Tensor): Derivative of the state.
            y (Tensor): Output tensor.
        """

        dx_lin, y_lin = self.linmod(u, x)
        fx = self.fxu(torch.cat([x,u], dim=1))
        if self.out_eq_nl:
            hx = self.hx(x)
            y = y_lin + hx
        else:
            y = y_lin
        
        return dx_lin + fx, y
        
    def _frame(self) -> Tuple[Tensor, ...]:
        """ 
            Only the linear coponent is parameterized, it is called inside its own forward method.
        """
        return self.linmod._frame()
    
    def _right_inverse(self, *args, **kwargs):
        pass

    def init_weights_(self, A0, B0, C0, requires_grad=True, init_type: str='linear', margin=1e-1):
        alpha = get_lyap_exp(A0) - margin
        if isinstance(self.linmod, SSLinear):
            self.linmod.init_weights_(A0, B0, C0, requires_grad)
        elif isinstance(self.linmod, L2BoundedLinear):
            _, gamma, _ = HInfCont.solve(A0, B0, C0, torch.zeros(self.ny, self.nu), alpha)
            self.linmod.init_weights_(A0, B0, C0, gamma + margin, alpha)  # Slightly higher L2 gain for sability
            self.gamma = self.linmod.gamma
        elif isinstance(self.linmod, H2Linear):
            _, gamma2, _ = H2Cont.solve(A0, B0, C0)
            self.linmod.init_weights_(A0, B0, C0, gamma2 + margin)
            self.gamma2 = self.linmod.gamma2

        self.alpha = self.linmod.alpha
        
        # Nonlinear part initialization
        if init_type == 'linear':
            self.fxu.layers.output.weight.data.zero_()
            if self.out_eq_nl:
                self.hx.layers.output.weight.data.zero_()

    def clone(self) -> 'GNSSM':
        clone = GNSSM(
            nu=self.nu,
            ny=self.ny,
            nx=self.nx,
            hidden_layers=self.hidden_layers,
            act_f=self.act_f_str,
            out_eq_nl=self.out_eq_nl,
            alpha=self.alpha,
            lin_model_type=self.lin_model_type,
            bias=self.bias,
            lip=self.lip,
            param=self.param,
            gamma=self.gamma,
            gamma2=self.gamma2
        )
        state_dict = self.state_dict()
        clone.load_state_dict(state_dict=state_dict)
        return clone

    def to_lure(self) -> Tuple[Tensor, ...]:
        """
        Returns the Lur'e representation of the GNSSM:
        ẋ = Ax + B2 u + B1 φ(q)
        y  = C2 x + D22 u + D21 φ(q)
        z  = C1 x + D12 u + D11 φ(q)
        where φ(q) is the stacked nonlinear activation vector.
        """
        A, B2, C2 = self.linmod._frame()

        # Get nonlinear input-output dimensions

        list_shapes_x, nq_x, weights_x = self._build_dims_weights(self.fxu)

        if self.out_eq_nl:
            list_shapes_y, nq_y, weights_y = self._build_dims_weights(self.hx)
        else:
            list_shapes_y, nq_y, weights_y = None, None, None

        
        B1 = self._build_B1(nq_x, list_shapes_x, weights_x, nq_y)
        C1, D12 = self._build_C1_D12(nq_x, list_shapes_x, weights_x, nq_y, list_shapes_y, weights_y)
        D11 = self._build_D11(nq_x, list_shapes_x, weights_x, nq_y, list_shapes_y, weights_y)
        D21 = self._build_D21(nq_y, list_shapes_y, nq_x, weights_y)
        D22 = torch.zeros(self.ny, self.nu)

        return A, B1, B2, C1, C2, D11, D12, D21, D22


    def _build_dims_weights(self, nl_part: FFNN | LBDN):
        list_shapes = []
        if isinstance(nl_part, FFNN):
            weights = nl_part.get_weights()
        else:
            weights = nl_part.extractWeightsSandwich()
        for layer in weights:
            list_shapes.append(layer.shape)
        nq = sum(shape[0] for shape in list_shapes[:-1])
        return list_shapes, nq, weights

    def _build_C1_D12(self, nq_x, list_shapes_x, weights_x, 
                      nq_y=None, list_shapes_y=None, weights_y=None):
        C1_x = torch.zeros((nq_x, self.nx))
        D12_x = torch.zeros((nq_x, self.nu))
        nh1, _ = list_shapes_x[0]
        C1_x[:nh1, :] = weights_x[0][:, :self.nx]
        D12_x[:nh1, :] = weights_x[0][:, self.nx:]
        if nq_y is not None and list_shapes_y is not None and weights_y is not None:
            C1_y = torch.zeros((nq_y, self.nx))
            D12_y = torch.zeros((nq_y, self.nu))
            nh1y, _ = list_shapes_y[0]
            C1_y[:nh1y, :] = weights_y[0][:, :self.nx]
            C1 = torch.cat((C1_x, C1_y), dim=0)
            D12 = torch.cat((D12_x, D12_y), dim=0)
        else:
            C1 = C1_x
            D12 = D12_x

        return C1, D12

    def _build_B1(self, nq_x, list_shapes_x, weights_x, nq_y):
        B1_x = torch.zeros((self.nx, nq_x))
        _, nhl = list_shapes_x[-1]
        B1_x[:, nq_x - nhl:] = weights_x[-1]

        B1_y = torch.zeros((self.nx, nq_y))
        B1 = torch.cat([B1_x, B1_y], dim=1)
        return B1
    
    def _build_D21(self, nq_y, list_shapes_y, nq_x, weights_y=None):
        D21 = torch.zeros((self.ny, nq_x + nq_y))
        if weights_y is not None:
            _, nhl = list_shapes_y[-1]
            D21[:, (nq_x + nq_y) - nhl:] = weights_y[-1]
        return D21

    def _build_D11(self, nq_x, list_shapes_x, weights_x, nq_y, list_shapes_y=None, weights_y=None):
        list_shapes_inter = list_shapes_x[1:-1]
        D11_x = torch.zeros((nq_x, nq_x))
        index_row, index_col = list_shapes_x[0][0], 0
        for i, shape in enumerate(list_shapes_inter):
            end_row = index_row + shape[0]
            end_col = index_col + shape[1]
            D11_x[index_row:end_row,
                  index_col:end_col] = weights_x[i + 1]
            index_row = end_row
            index_col = end_col

        D11_y = torch.zeros((nq_y, nq_y))
        if list_shapes_y is not None and weights_y is not None:
            index_row, index_col = list_shapes_y[0][0], 0
            for i, shape in enumerate(list_shapes_inter):
                end_row = index_row + shape[0]
                end_col = index_col + shape[1]
                D11_y[index_row:end_row,
                      index_col:end_col] = weights_y[i + 1]
                index_row = end_row
                index_col = end_col
        D11 = torch.block_diag(D11_x, D11_y)
        return D11
    
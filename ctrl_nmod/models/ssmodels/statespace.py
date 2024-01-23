'''
    This module includes some standard state-space arhcitectures
'''
from torch.nn import Module, Linear
from torch.nn.parameter import Parameter


class NNLinear(Module):
    """
        neural network module corresponding to a linear state-space model
            x^+ = Ax + Bu
            y = Cx
    """
    def __init__(self, input_dim: int, output_dim: int, config) -> None:
        super(NNLinear, self).__init__()

        self.input_dim = input_dim
        self.state_dim = config.nx
        self.output_dim = output_dim

        self.A = Linear(self.state_dim, self.state_dim, bias=False)
        self.B = Linear(self.input_dim, self.state_dim, bias=False)
        self.C = Linear(self.state_dim, self.output_dim, bias=False)
        self.config = config

    def forward(self, u, x):
        dx = self.A(x) + self.B(u)
        y = self.C(x)

        return dx, y

    def init_weights_(self, A0, B0, C0, is_grad=True):
        self.A.weight = Parameter(A0)
        self.B.weight = Parameter(B0)
        self.C.weight = Parameter(C0)
        if is_grad is False:
            self.A.requires_grad_(False)
            self.B.requires_grad_(False)
            self.C.requires_grad_(False)

    def clone(self):  # Method called by the simulator
        copy = type(self)(self.input_dim, self.output_dim, self.config)
        copy.load_state_dict(self.state_dict())
        return copy
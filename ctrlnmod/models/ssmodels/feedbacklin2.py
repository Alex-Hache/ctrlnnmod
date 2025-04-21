import torch.nn as nn
from ..feedforward.linearizers import OutputFeedbackLinearizingcontroller, StateFeedbackLinearizingcontroller
from ctrlnmod.utils import FrameCacheManager
from .base import SSModel
from .linear import SSLinear
import torch


class FLNSSM(SSModel):
    def __init__(self, nu: int, ny: int, nx: int, linear_model: SSLinear, linearizer: OutputFeedbackLinearizingcontroller|StateFeedbackLinearizingcontroller, 
                 nd: int = 0):
        super(FLNSSM, self).__init__()

        self.nu = nu
        self.ny = ny
        self.nx = nx
        self.linear_model = linear_model
        self.linearizer = linearizer
        self.nd = nd

        self._frame_cache = FrameCacheManager()


    def forward(self, u, v):
        pass


class OFLNSSM(FLNSSM):
    """
        Base constructor for Output Feedback Linearizable Neural State-Space Model :
            \dot{x} = Ax + B\beta(y, d)(u + \alpha(y, d)) + Gd
            y = Cx
        Args:
            nu (int): system's number of inputs
            ny (int): systems's number of measured outputs
            nx (int): system's order
            linear_model (nn.Module): Linear part of the model
            linearizer (nn.Module): Linearizing controller
            nd (int): number of measured disturbances to absorb by the linearizing controller
                Defaults to 0
    """
    def __init__(self, nu: int, ny: int, nx: int, linear_model: SSLinear, linearizer: OutputFeedbackLinearizingcontroller, nd = 0):
        super().__init__(nu, ny, nx, linear_model, linearizer, nd)

        assert self.ny + self.nd == linearizer.n_inputs

    def forward(self, u, y, x, d=None):
        
        v = self.linearizer(y, u, d)
        dx, y = self.linear_model(v, x, d)
        return dx, y

    
    def _frame(self):
        if self._frame_cache.is_caching and self._frame_cache.cache is not None:
            return self._frame_cache.cache
        
        A, B, C = self.linear_model._frame()
        if self._frame_cache.is_caching:
            self._frame_cache.cache = (A, B, C, torch.zeros((self.ny, self.nu)))
        
        return A, B, C, torch.zeros((self.ny, self.nu))
    
    def init_weights_(self, linear_weights: dict, linearizer: dict):
        self.linear_model.init_weights_(linear_weights)
        #self.linearizer.init_weights_(linearizer)
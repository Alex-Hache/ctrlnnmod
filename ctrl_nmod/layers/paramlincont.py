'''
    This module implements several parameterizations for continuous neural state-space models
    it includes :
        - Alpha stable linear models
        - L2 gain bounded linear models -- H inifinty norm
        - QSR dissipative linear models including :
            - L2 gain bounded
        - H2 gain
        - Incrementally QSR dissipative models including :
            - Incremental L2 gain bounded
            - Incrementally input passive
            - Incrementally output passive
'''

from torch.nn import Module


class ContinuousLinearSS(Module):
    '''
        This is base class for continuous RNN for which weights
        can be written as a linear state-space model
    '''
    def __init__(self, nu: int, ny: int, nx: int) -> None:
        super(ContinuousLinearSS, self).__init__()
        
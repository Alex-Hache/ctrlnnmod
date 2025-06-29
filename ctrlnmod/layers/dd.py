import torch
import torch.nn as nn
from torch.linalg import inv, cholesky


class DDLayer(nn.Module):
    def __init__(self, Ui: torch.Tensor) -> None:
        super(DDLayer, self).__init__()
        """
            Ui : inverse of the upper Cholesky decomposition associated to the DD problem
        """
        self.Ui = Ui
        self.act = nn.ReLU()

    def forward(self, M):
        """
        M : linear matrix inequality in Sn+
        """

        Q = self.Ui.T @ M @ self.Ui
        dQ = torch.diag(Q)
        delta_Q = self.act(torch.sum(torch.abs(Q), dim=1) - dQ - torch.abs(dQ))

        DQ = torch.diag(delta_Q)  # (Une) distance à l'ensemble DD+

        return DQ

    def updateU_(self, M):
        """
        From the current M value we update U to update search region in DD+
        If correct : DDLayer(updateU_(M)) = I
        """
        # Q = self(M)
        # M_next = inv(self.Ui.T) @ Q @ inv(self.Ui)
        # assert torch.all(M_next == M)
        self.Ui = inv(cholesky(M).mH)

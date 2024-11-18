import torch
import torch.nn as nn
from typing import Optional


class ExponentialSimilarity(nn.Module):

    def __init__(
        self, starting_value: Optional[float] = None, requires_grad: bool = True
    ) -> None:
        """The exponential similarity function: e^(-k*|x|). Given that the k is an
        optional learnable parameter.

        Args:
            starting_value (Optional[float], optional): The starting value of k.
                If None defaults to 1.
            requires_grad (bool, optional): Whether k should be a
                learnable parameter. Defaults to True.
        """
        super().__init__()
        self.param = nn.Parameter(
            torch.Tensor([1] if starting_value is None else [starting_value]),
            requires_grad=requires_grad,
        )

    def forward(self, x):
        return torch.exp(-self.param * x.abs())


class AbsoluteInverseSimilarity(nn.Module):

    def __init__(
        self, starting_value: Optional[float] = None, requires_grad: bool = False
    ) -> None:
        """The absolute inverse similarity function: 1 / (1 + k*|x|). Given that
        the k is an optional learnable parameter.

        Args:
            starting_value (Optional[float], optional): The starting value of k.
                If None, defaults to 1.
            requires_grad (bool, optional): Whether k should be a
                learnable parameter. Defaults to False.
        """
        super().__init__()
        self.param = nn.Parameter(
            torch.Tensor([1] if starting_value is None else [starting_value]),
            requires_grad=requires_grad,
        )

    def forward(self, x):
        return 1 / (1 + self.param * x.abs())

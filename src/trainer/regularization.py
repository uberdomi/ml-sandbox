"""This module contains the Regularization class, which implements various regularization techniques for PyTorch models."""

from typing import Literal, List

class Regularizer:
    """Regularization techniques for PyTorch models."""
    model_params = [] # Placeholder for model parameters
    
    def __init__(
        self,
        method: Literal["L1","L2","Elastic"] = 'L1',
        weight: float = 0.1,
        weight2: float = 0.1,
    ):
        """
        Initialize the regularizer with the model and weight decay factor.
        
        Args:
            method: Method used for regularization. Supported are "L1", "L2" and "Elastic".
            weight: Weight for the penalty term.
            weight2: Weight for the l2 regression in case of the elastic regularization.
        """
        self.method = method
        self.weight = weight
        self.weight2 = weight2

    def calculate_sum_weights(self, p: int) -> float:
        """
        Calculates the sum of Lp norms to the power p of all weight matrices.
        Args:
            p: Power factor.
        Returns:
            Sum of all Lp norms to the power of p.
        """
        # Don't include NormLayers and biases in the regularization
        return sum(param.abs().pow(p).sum() for name, param in self.model_params
                    if "bias" not in name and "norm" not in name)

    def apply_regularization(self, named_model_parameters: List) -> float:
        """
        Calculates the proper regularization terms for the weight matrices.

        Returns:
            The regularization term to be added to the loss function.
        """
        self.model_params = named_model_parameters
        
        if self.method == "L1":
            sum_params = self.calculate_sum_weights(1)
            
            return self.weight * sum_params
        elif self.method == "L2":
            sum_params = self.calculate_sum_weights(2)
            
            return self.weight * sum_params
        elif self.method == "Elastic":
            sum_params1 = self.calculate_sum_weights(1)
            sum_params2 = self.calculate_sum_weights(2)
            
            return self.weight * sum_params1 + self.weight2 * sum_params2
        else:
            return 0.0
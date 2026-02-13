"""
Uncertainty-aware optimization module.

This module implements optimization strategies that leverage uncertainty estimates:
- Bayesian optimization with acquisition functions
- Uncertainty-weighted loss functions
- Active learning strategies for data-efficient training
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Tuple, Dict
from scipy.stats import norm


class UncertaintyAwareOptimizer:
    """
    Optimizer that leverages uncertainty estimates for improved training.
    
    Implements uncertainty-weighted loss and active learning strategies.
    """
    
    def __init__(
        self,
        model: nn.Module,
        base_optimizer: torch.optim.Optimizer,
        uncertainty_weight: float = 0.1,
        use_epistemic: bool = True,
        use_aleatoric: bool = True
    ):
        """
        Initialize uncertainty-aware optimizer.
        
        Args:
            model: The neural network model
            base_optimizer: Base PyTorch optimizer (e.g., Adam, SGD)
            uncertainty_weight: Weight for uncertainty regularization
            use_epistemic: Whether to use epistemic uncertainty
            use_aleatoric: Whether to use aleatoric uncertainty
        """
        self.model = model
        self.optimizer = base_optimizer
        self.uncertainty_weight = uncertainty_weight
        self.use_epistemic = use_epistemic
        self.use_aleatoric = use_aleatoric
        
    def step(
        self,
        loss: torch.Tensor,
        epistemic_uncertainty: Optional[torch.Tensor] = None,
        aleatoric_uncertainty: Optional[torch.Tensor] = None
    ) -> float:
        """
        Perform optimization step with uncertainty weighting.
        
        Args:
            loss: Base loss value
            epistemic_uncertainty: Epistemic (model) uncertainty
            aleatoric_uncertainty: Aleatoric (data) uncertainty
            
        Returns:
            Total loss value
        """
        total_loss = loss
        
        # Add uncertainty regularization
        if self.use_epistemic and epistemic_uncertainty is not None:
            # Penalize high epistemic uncertainty
            epistemic_penalty = self.uncertainty_weight * torch.mean(epistemic_uncertainty)
            total_loss = total_loss + epistemic_penalty
        
        if self.use_aleatoric and aleatoric_uncertainty is not None:
            # Aleatoric uncertainty is already in the loss through log_var
            pass
        
        # Backpropagation
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()


class BayesianOptimizer:
    """
    Bayesian Optimization for hyperparameter tuning.
    
    Uses Gaussian Process surrogate model with acquisition functions.
    """
    
    def __init__(
        self,
        bounds: np.ndarray,
        acquisition_fn: str = 'ei',
        xi: float = 0.01,
        kappa: float = 2.576
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            bounds: Parameter bounds as array of shape (n_params, 2)
            acquisition_fn: Acquisition function ('ei', 'ucb', 'poi')
            xi: Exploration parameter for EI
            kappa: Exploration parameter for UCB
        """
        self.bounds = bounds
        self.acquisition_fn = acquisition_fn
        self.xi = xi
        self.kappa = kappa
        
        self.X_samples = []
        self.y_samples = []
    
    def expected_improvement(
        self,
        X: np.ndarray,
        X_samples: np.ndarray,
        y_samples: np.ndarray,
        gp_mean: np.ndarray,
        gp_std: np.ndarray
    ) -> np.ndarray:
        """
        Compute Expected Improvement acquisition function.
        
        Args:
            X: Points at which to evaluate
            X_samples: Observed points
            y_samples: Observed values
            gp_mean: GP mean predictions
            gp_std: GP standard deviation predictions
            
        Returns:
            Expected improvement values
        """
        y_best = np.max(y_samples)
        
        with np.errstate(divide='warn'):
            improvement = gp_mean - y_best - self.xi
            Z = improvement / (gp_std + 1e-9)
            ei = improvement * norm.cdf(Z) + gp_std * norm.pdf(Z)
            ei[gp_std == 0.0] = 0.0
        
        return ei
    
    def upper_confidence_bound(
        self,
        gp_mean: np.ndarray,
        gp_std: np.ndarray
    ) -> np.ndarray:
        """
        Compute Upper Confidence Bound acquisition function.
        
        Args:
            gp_mean: GP mean predictions
            gp_std: GP standard deviation predictions
            
        Returns:
            UCB values
        """
        return gp_mean + self.kappa * gp_std
    
    def propose_location(self) -> np.ndarray:
        """
        Propose next evaluation point using acquisition function.
        
        Returns:
            Proposed parameter values
        """
        # This is a simplified version
        # In practice, would use scipy.optimize or similar
        n_random = 1000
        n_dims = self.bounds.shape[0]
        
        # Random sampling within bounds
        X_random = np.random.uniform(
            self.bounds[:, 0],
            self.bounds[:, 1],
            size=(n_random, n_dims)
        )
        
        # Would compute GP predictions here
        # For now, return random point
        return X_random[0]


class ActiveLearner:
    """
    Active learning strategy for data-efficient model training.
    
    Selects informative samples based on uncertainty for labeling.
    """
    
    def __init__(
        self,
        strategy: str = 'uncertainty',
        batch_size: int = 10
    ):
        """
        Initialize active learner.
        
        Args:
            strategy: Selection strategy ('uncertainty', 'margin', 'entropy')
            batch_size: Number of samples to select per iteration
        """
        self.strategy = strategy
        self.batch_size = batch_size
    
    def select_samples(
        self,
        uncertainties: np.ndarray,
        n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Select most informative samples based on uncertainty.
        
        Args:
            uncertainties: Uncertainty estimates for each sample
            n_samples: Number of samples to select (defaults to batch_size)
            
        Returns:
            Indices of selected samples
        """
        if n_samples is None:
            n_samples = self.batch_size
        
        # Select samples with highest uncertainty
        if self.strategy == 'uncertainty':
            indices = np.argsort(uncertainties)[-n_samples:]
        else:
            # Other strategies can be implemented
            indices = np.argsort(uncertainties)[-n_samples:]
        
        return indices
    
    def compute_entropy(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Compute prediction entropy.
        
        Args:
            probabilities: Probability predictions
            
        Returns:
            Entropy values
        """
        entropy = -np.sum(probabilities * np.log(probabilities + 1e-9), axis=1)
        return entropy


class UncertaintyWeightedLoss(nn.Module):
    """
    Loss function that weights samples by their uncertainty.
    """
    
    def __init__(
        self,
        base_loss: nn.Module,
        uncertainty_type: str = 'epistemic',
        weight_scheme: str = 'inverse'
    ):
        """
        Initialize uncertainty-weighted loss.
        
        Args:
            base_loss: Base loss function (e.g., MSELoss)
            uncertainty_type: Type of uncertainty to use
            weight_scheme: Weighting scheme ('inverse', 'exponential')
        """
        super().__init__()
        self.base_loss = base_loss
        self.uncertainty_type = uncertainty_type
        self.weight_scheme = weight_scheme
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        uncertainties: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted loss.
        
        Args:
            predictions: Model predictions
            targets: Ground truth values
            uncertainties: Uncertainty estimates
            
        Returns:
            Weighted loss
        """
        # Compute base loss
        losses = self.base_loss(predictions, targets)
        
        # Compute weights from uncertainties
        if self.weight_scheme == 'inverse':
            weights = 1.0 / (uncertainties + 1e-8)
        elif self.weight_scheme == 'exponential':
            weights = torch.exp(-uncertainties)
        else:
            weights = torch.ones_like(uncertainties)
        
        # Normalize weights
        weights = weights / torch.sum(weights)
        
        # Weighted loss
        weighted_loss = torch.sum(weights * losses)
        
        return weighted_loss

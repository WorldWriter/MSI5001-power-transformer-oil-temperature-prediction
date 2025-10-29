"""
PyTorch implementation of MLP Regressor with GPU support.

This module provides a drop-in replacement for sklearn.neural_network.MLPRegressor
with GPU acceleration support.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


def get_device() -> torch.device:
    """
    Automatically detect and return the best available device.

    Priority: CUDA > MPS (Apple Silicon) > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    else:
        device = torch.device("cpu")
        print("Using CPU (no GPU available)")
    return device


class MLPNetwork(nn.Module):
    """
    Multi-layer Perceptron neural network for regression.
    """

    def __init__(self, input_size: int, hidden_layer_sizes: Tuple[int, ...], dropout: float = 0.0):
        super().__init__()
        layers = []
        prev_size = input_size

        for hidden_size in hidden_layer_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_size = hidden_size

        # Output layer (single value for regression)
        layers.append(nn.Linear(prev_size, 1))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


class PyTorchMLPRegressor:
    """
    PyTorch-based MLP Regressor with scikit-learn compatible interface.

    This class mimics the sklearn.neural_network.MLPRegressor API to allow
    drop-in replacement with GPU acceleration.

    Parameters
    ----------
    hidden_layer_sizes : tuple of int, default=(100,)
        The number of neurons in each hidden layer.
    activation : str, default='relu'
        Activation function (only 'relu' is supported currently).
    alpha : float, default=1e-4
        L2 regularization term (weight decay).
    learning_rate_init : float, default=1e-3
        Initial learning rate for optimizer.
    max_iter : int, default=200
        Maximum number of training epochs.
    batch_size : int, default=32
        Size of mini-batches for training.
    random_state : int, default=None
        Random seed for reproducibility.
    early_stopping : bool, default=False
        Whether to use early stopping based on validation loss.
    validation_fraction : float, default=0.1
        Fraction of training data to use as validation set for early stopping.
    n_iter_no_change : int, default=10
        Number of epochs with no improvement to wait before stopping.
    tol : float, default=1e-4
        Tolerance for early stopping improvement.
    verbose : bool, default=False
        Whether to print training progress.
    device : str or torch.device, default='auto'
        Device to use for training ('auto', 'cuda', 'mps', or 'cpu').
    """

    def __init__(
        self,
        hidden_layer_sizes: Tuple[int, ...] = (100,),
        activation: str = "relu",
        alpha: float = 1e-4,
        learning_rate_init: float = 1e-3,
        max_iter: int = 200,
        batch_size: int = 32,
        random_state: int | None = None,
        early_stopping: bool = False,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
        tol: float = 1e-4,
        verbose: bool = False,
        device: str | torch.device = "auto",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.alpha = alpha
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.batch_size = batch_size
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.tol = tol
        self.verbose = verbose
        self.device_str = device

        # Set random seeds
        if random_state is not None:
            torch.manual_seed(random_state)
            np.random.seed(random_state)

        # Model will be initialized during fit
        self.model_ = None
        self.device_ = None
        self.n_features_in_ = None
        self.n_iter_ = 0
        self.best_loss_ = float('inf')
        self.train_time_ = 0.0

    def _get_device(self) -> torch.device:
        """Get the device to use for training."""
        if self.device_str == "auto":
            return get_device()
        else:
            return torch.device(self.device_str)

    def _prepare_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[DataLoader, DataLoader | None]:
        """
        Convert numpy arrays to PyTorch DataLoader.

        If early_stopping is enabled, split into train and validation sets.
        """
        X_tensor = torch.from_numpy(X.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.float32))

        if self.early_stopping:
            # Split into train and validation
            n_samples = len(X)
            n_val = int(n_samples * self.validation_fraction)

            # Shuffle indices
            if self.random_state is not None:
                rng = np.random.RandomState(self.random_state)
            else:
                rng = np.random.RandomState()
            indices = rng.permutation(n_samples)

            train_indices = indices[n_val:]
            val_indices = indices[:n_val]

            train_dataset = TensorDataset(X_tensor[train_indices], y_tensor[train_indices])
            val_dataset = TensorDataset(X_tensor[val_indices], y_tensor[val_indices])

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

            return train_loader, val_loader
        else:
            dataset = TensorDataset(X_tensor, y_tensor)
            train_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
            return train_loader, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the MLP model to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : PyTorchMLPRegressor
            Returns self for method chaining.
        """
        start_time = time.time()

        # Initialize device
        self.device_ = self._get_device()

        # Store input shape
        self.n_features_in_ = X.shape[1]

        # Create model
        self.model_ = MLPNetwork(
            input_size=self.n_features_in_,
            hidden_layer_sizes=self.hidden_layer_sizes,
            dropout=0.0  # Can add dropout if needed
        ).to(self.device_)

        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate_init,
            weight_decay=self.alpha
        )
        criterion = nn.MSELoss()

        # Prepare data
        train_loader, val_loader = self._prepare_data(X, y)

        # Training loop
        best_val_loss = float('inf')
        epochs_no_improve = 0

        for epoch in range(self.max_iter):
            # Training phase
            self.model_.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X = batch_X.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()
                outputs = self.model_(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_X)

            train_loss /= len(train_loader.dataset)

            # Validation phase (if early stopping enabled)
            if self.early_stopping and val_loader is not None:
                self.model_.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(self.device_)
                        batch_y = batch_y.to(self.device_)
                        outputs = self.model_(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * len(batch_X)

                val_loss /= len(val_loader.dataset)

                # Check for improvement
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save best model state
                    self.best_state_ = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    epochs_no_improve += 1

                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.max_iter} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

                # Early stopping check
                if epochs_no_improve >= self.n_iter_no_change:
                    if self.verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    # Restore best model
                    self.model_.load_state_dict(self.best_state_)
                    self.n_iter_ = epoch + 1
                    break
            else:
                if self.verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.max_iter} - Train Loss: {train_loss:.6f}")

            self.n_iter_ = epoch + 1

        self.train_time_ = time.time() - start_time

        if self.verbose:
            print(f"Training completed in {self.train_time_:.2f}s ({self.n_iter_} epochs)")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the trained model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        self.model_.eval()

        X_tensor = torch.from_numpy(X.astype(np.float32)).to(self.device_)

        with torch.no_grad():
            predictions = self.model_(X_tensor)

        return predictions.cpu().numpy()

    def save(self, path: str | Path):
        """Save model to disk."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before saving")

        save_dict = {
            'model_state_dict': self.model_.state_dict(),
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'n_features_in': self.n_features_in_,
            'n_iter': self.n_iter_,
        }
        torch.save(save_dict, path)

    def load(self, path: str | Path):
        """Load model from disk."""
        save_dict = torch.load(path, map_location='cpu')

        self.hidden_layer_sizes = save_dict['hidden_layer_sizes']
        self.n_features_in_ = save_dict['n_features_in']
        self.n_iter_ = save_dict['n_iter']

        self.device_ = self._get_device()
        self.model_ = MLPNetwork(
            input_size=self.n_features_in_,
            hidden_layer_sizes=self.hidden_layer_sizes,
        ).to(self.device_)

        self.model_.load_state_dict(save_dict['model_state_dict'])
        self.model_.eval()

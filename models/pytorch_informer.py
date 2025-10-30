"""
PyTorch implementation of Informer Regressor with GPU support.

This module provides a drop-in replacement for traditional regression models
with Informer architecture for time series prediction with GPU acceleration.

Based on: https://github.com/zhouhaoyi/Informer2020
"""

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Import Informer from local copy in models/informer_arch/
# This avoids path conflicts with the project's models directory
try:
    from models.informer_arch.model import Informer
except ImportError as e:
    raise ImportError(
        f"Failed to import Informer: {e}\n"
        f"Make sure models/informer_arch/ contains the Informer model files."
    )


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


class PyTorchInformerRegressor:
    """
    PyTorch-based Informer Regressor with scikit-learn compatible interface.

    This class provides a scikit-learn compatible API for Informer-based regression
    with GPU acceleration support.

    Parameters
    ----------
    seq_len : int, default=96
        Input sequence length (encoder)
    label_len : int, default=48
        Start token length for decoder
    pred_len : int, default=24
        Prediction sequence length
    d_model : int, default=512
        Dimension of model embeddings
    n_heads : int, default=8
        Number of attention heads
    e_layers : int, default=2
        Number of encoder layers
    d_layers : int, default=1
        Number of decoder layers
    d_ff : int, default=2048
        Dimension of feedforward network
    factor : int, default=5
        ProbSparse attn factor
    dropout : float, default=0.05
        Dropout rate
    attn : str, default='prob'
        Attention mechanism ('prob' or 'full')
    activation : str, default='gelu'
        Activation function
    learning_rate_init : float, default=1e-4
        Initial learning rate for optimizer
    max_iter : int, default=10
        Maximum number of training epochs
    batch_size : int, default=32
        Size of mini-batches for training
    random_state : int, default=None
        Random seed for reproducibility
    early_stopping : bool, default=True
        Whether to use early stopping based on validation loss
    validation_fraction : float, default=0.1
        Fraction of training data to use as validation set for early stopping
    n_iter_no_change : int, default=5
        Number of epochs with no improvement to wait before stopping
    tol : float, default=1e-4
        Tolerance for early stopping improvement
    verbose : bool, default=False
        Whether to print training progress
    device : str or torch.device, default='auto'
        Device to use for training ('auto', 'cuda', 'mps', or 'cpu')
    """

    def __init__(
        self,
        seq_len: int = 96,
        label_len: int = 48,
        pred_len: int = 24,
        d_model: int = 512,
        n_heads: int = 8,
        e_layers: int = 2,
        d_layers: int = 1,
        d_ff: int = 2048,
        factor: int = 5,
        dropout: float = 0.05,
        attn: str = 'prob',
        activation: str = 'gelu',
        learning_rate_init: float = 1e-4,
        max_iter: int = 10,
        batch_size: int = 32,
        random_state: int | None = None,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 5,
        tol: float = 1e-4,
        verbose: bool = False,
        device: str | torch.device = "auto",
    ):
        # Informer architecture parameters
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_layers = d_layers
        self.d_ff = d_ff
        self.factor = factor
        self.dropout = dropout
        self.attn = attn
        self.activation = activation

        # Training parameters
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

    def _prepare_informer_inputs(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Prepare inputs for Informer model from standard 3D sequences.

        For simplification in this fast prototype:
        - We extract/construct 4 time features for TimeFeatureEmbedding (freq='h')
        - Decoder input is created by zero-padding

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, seq_len, n_features)
        y : np.ndarray, optional
            Target values of shape (n_samples,)

        Returns
        -------
        x_enc : Encoder input (batch, seq_len, n_features)
        x_mark_enc : Encoder time marks (batch, seq_len, 4) - time features only
        x_dec : Decoder input (batch, label_len + pred_len, n_features)
        x_mark_dec : Decoder time marks (batch, label_len + pred_len, 4) - time features only
        y_tensor : Target tensor (optional)
        """
        batch_size, actual_seq_len, n_features = X.shape

        # Encoder inputs: use the full input sequence
        x_enc = torch.from_numpy(X.astype(np.float32))

        # For time marks, we use normalized sin/cos time features if available
        # TimeFeatureEmbedding for freq='h' expects 4 features
        # We'll use the first 4 sin/cos encoded features: hour_sin, hour_cos, dow_sin, dow_cos
        # Feature indices based on experiment_utils.py:
        # 0-1: HULL, MULL; 2-5: hour, dayofweek, month, day_of_year
        # 6-7: hour_sin, hour_cos; 8-9: dow_sin, dow_cos
        if n_features >= 10:
            # Extract sin/cos time features (indices 6-9, which is 6:10 in Python slicing)
            time_features = X[:, :, 6:10]  # hour_sin, hour_cos, dow_sin, dow_cos
        else:
            # Fallback: create dummy time features (all zeros)
            time_features = np.zeros((batch_size, actual_seq_len, 4), dtype=np.float32)

        x_mark_enc = torch.from_numpy(time_features.astype(np.float32))

        # Decoder inputs:
        # - First label_len steps: use last label_len steps from encoder
        # - Next pred_len steps: zeros (to be predicted)
        x_dec = torch.zeros((batch_size, self.label_len + self.pred_len, n_features), dtype=torch.float32)

        # Copy last label_len steps from encoder to decoder start
        if actual_seq_len >= self.label_len:
            x_dec[:, :self.label_len, :] = x_enc[:, -self.label_len:, :]
        else:
            # If sequence is shorter, use what we have
            x_dec[:, :actual_seq_len, :] = x_enc

        # Decoder time marks: similarly extract time features
        x_mark_dec = torch.zeros((batch_size, self.label_len + self.pred_len, 4), dtype=torch.float32)
        if actual_seq_len >= self.label_len and n_features >= 10:
            # Copy time features from last label_len steps
            x_mark_dec[:, :self.label_len, :] = x_mark_enc[:, -self.label_len:, :]

        # Target tensor
        y_tensor = None
        if y is not None:
            y_tensor = torch.from_numpy(y.astype(np.float32))

        return x_enc, x_mark_enc, x_dec, x_mark_dec, y_tensor

    def _prepare_data(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[DataLoader, DataLoader | None]:
        """
        Convert numpy arrays to PyTorch DataLoader.

        If early_stopping is enabled, split into train and validation sets.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_samples, seq_length, n_features)
        y : np.ndarray
            Target values of shape (n_samples,)
        """
        # Validate input shape
        if X.ndim != 3:
            raise ValueError(
                f"Informer expects 3D input (n_samples, seq_length, n_features), "
                f"got shape {X.shape}. "
                f"Make sure to use create_sliding_windows_for_rnn() instead of "
                f"create_sliding_windows() for Informer models."
            )

        # Prepare Informer-specific inputs
        x_enc, x_mark_enc, x_dec, x_mark_dec, y_tensor = self._prepare_informer_inputs(X, y)

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

            train_dataset = TensorDataset(
                x_enc[train_indices],
                x_mark_enc[train_indices],
                x_dec[train_indices],
                x_mark_dec[train_indices],
                y_tensor[train_indices]
            )
            val_dataset = TensorDataset(
                x_enc[val_indices],
                x_mark_enc[val_indices],
                x_dec[val_indices],
                x_mark_dec[val_indices],
                y_tensor[val_indices]
            )

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True
            )
            val_loader = DataLoader(
                val_dataset, batch_size=self.batch_size, shuffle=False
            )

            return train_loader, val_loader
        else:
            dataset = TensorDataset(x_enc, x_mark_enc, x_dec, x_mark_dec, y_tensor)
            train_loader = DataLoader(
                dataset, batch_size=self.batch_size, shuffle=True
            )
            return train_loader, None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the Informer model to training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, seq_length, n_features)
            Training data (3D array for sequences)
        y : np.ndarray of shape (n_samples,)
            Target values

        Returns
        -------
        self : PyTorchInformerRegressor
            Returns self for method chaining
        """
        start_time = time.time()

        # Initialize device
        self.device_ = self._get_device()

        # Store input shape
        if X.ndim != 3:
            raise ValueError(
                f"Informer expects 3D input (n_samples, seq_length, n_features), "
                f"got shape {X.shape}"
            )

        actual_seq_len = X.shape[1]
        self.n_features_in_ = X.shape[2]

        # Adjust seq_len if input is smaller
        if actual_seq_len < self.seq_len:
            print(f"Warning: Input sequence length ({actual_seq_len}) is smaller than "
                  f"configured seq_len ({self.seq_len}). Adjusting seq_len to {actual_seq_len}.")
            self.seq_len = actual_seq_len
            if self.label_len > actual_seq_len:
                self.label_len = actual_seq_len // 2

        # Create Informer model
        # Note: For single-step regression, pred_len should be 1
        # But we keep it configurable for future multi-step forecasting
        self.model_ = Informer(
            enc_in=self.n_features_in_,
            dec_in=self.n_features_in_,
            c_out=1,  # Single output for regression
            seq_len=self.seq_len,
            label_len=self.label_len,
            out_len=1,  # Predict 1 step ahead
            factor=self.factor,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            attn=self.attn,
            embed='timeF',  # Use TimeFeatureEmbedding for simplicity
            freq='h',  # Hourly frequency
            activation=self.activation,
            output_attention=False,
            distil=True,
            mix=True,
            device=self.device_
        ).to(self.device_)

        # Setup optimizer and loss
        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.learning_rate_init
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

            for batch_data in train_loader:
                batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = batch_data

                batch_x_enc = batch_x_enc.to(self.device_)
                batch_x_mark_enc = batch_x_mark_enc.to(self.device_)
                batch_x_dec = batch_x_dec.to(self.device_)
                batch_x_mark_dec = batch_x_mark_dec.to(self.device_)
                batch_y = batch_y.to(self.device_)

                optimizer.zero_grad()

                # Forward pass
                outputs = self.model_(
                    batch_x_enc, batch_x_mark_enc,
                    batch_x_dec, batch_x_mark_dec
                )

                # outputs shape: (batch, pred_len, c_out) = (batch, 1, 1)
                # Squeeze to (batch,) for MSE loss
                outputs = outputs.squeeze(-1).squeeze(-1)

                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model_.parameters(), max_norm=1.0)

                optimizer.step()

                train_loss += loss.item() * len(batch_y)

            train_loss /= len(train_loader.dataset)

            # Validation phase (if early stopping enabled)
            if self.early_stopping and val_loader is not None:
                self.model_.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for batch_data in val_loader:
                        batch_x_enc, batch_x_mark_enc, batch_x_dec, batch_x_mark_dec, batch_y = batch_data

                        batch_x_enc = batch_x_enc.to(self.device_)
                        batch_x_mark_enc = batch_x_mark_enc.to(self.device_)
                        batch_x_dec = batch_x_dec.to(self.device_)
                        batch_x_mark_dec = batch_x_mark_dec.to(self.device_)
                        batch_y = batch_y.to(self.device_)

                        outputs = self.model_(
                            batch_x_enc, batch_x_mark_enc,
                            batch_x_dec, batch_x_mark_dec
                        )
                        outputs = outputs.squeeze(-1).squeeze(-1)

                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * len(batch_y)

                val_loss /= len(val_loader.dataset)

                # Check for improvement
                if val_loss < best_val_loss - self.tol:
                    best_val_loss = val_loss
                    epochs_no_improve = 0
                    # Save best model state
                    self.best_state_ = {k: v.cpu().clone() for k, v in self.model_.state_dict().items()}
                else:
                    epochs_no_improve += 1

                if self.verbose and (epoch + 1) % 2 == 0:
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
                if self.verbose and (epoch + 1) % 2 == 0:
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
        X : np.ndarray of shape (n_samples, seq_length, n_features)
            Input data (3D array for sequences)

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values
        """
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before calling predict()")

        if X.ndim != 3:
            raise ValueError(
                f"Informer expects 3D input (n_samples, seq_length, n_features), "
                f"got shape {X.shape}"
            )

        self.model_.eval()

        # Prepare Informer inputs
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = self._prepare_informer_inputs(X)

        x_enc = x_enc.to(self.device_)
        x_mark_enc = x_mark_enc.to(self.device_)
        x_dec = x_dec.to(self.device_)
        x_mark_dec = x_mark_dec.to(self.device_)

        with torch.no_grad():
            predictions = self.model_(x_enc, x_mark_enc, x_dec, x_mark_dec)
            # predictions shape: (batch, 1, 1), squeeze to (batch,)
            predictions = predictions.squeeze(-1).squeeze(-1)

        return predictions.cpu().numpy()

    def save(self, path: str | Path):
        """Save model to disk."""
        if self.model_ is None:
            raise RuntimeError("Model must be fitted before saving")

        save_dict = {
            'model_state_dict': self.model_.state_dict(),
            'seq_len': self.seq_len,
            'label_len': self.label_len,
            'pred_len': self.pred_len,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'e_layers': self.e_layers,
            'd_layers': self.d_layers,
            'd_ff': self.d_ff,
            'factor': self.factor,
            'dropout': self.dropout,
            'attn': self.attn,
            'activation': self.activation,
            'n_features_in': self.n_features_in_,
            'n_iter': self.n_iter_,
        }
        torch.save(save_dict, path)

    def load(self, path: str | Path):
        """Load model from disk."""
        save_dict = torch.load(path, map_location='cpu')

        # Restore parameters
        self.seq_len = save_dict['seq_len']
        self.label_len = save_dict['label_len']
        self.pred_len = save_dict['pred_len']
        self.d_model = save_dict['d_model']
        self.n_heads = save_dict['n_heads']
        self.e_layers = save_dict['e_layers']
        self.d_layers = save_dict['d_layers']
        self.d_ff = save_dict['d_ff']
        self.factor = save_dict['factor']
        self.dropout = save_dict['dropout']
        self.attn = save_dict['attn']
        self.activation = save_dict['activation']
        self.n_features_in_ = save_dict['n_features_in']
        self.n_iter_ = save_dict['n_iter']

        self.device_ = self._get_device()

        # Recreate model
        self.model_ = Informer(
            enc_in=self.n_features_in_,
            dec_in=self.n_features_in_,
            c_out=1,
            seq_len=self.seq_len,
            label_len=self.label_len,
            out_len=1,
            factor=self.factor,
            d_model=self.d_model,
            n_heads=self.n_heads,
            e_layers=self.e_layers,
            d_layers=self.d_layers,
            d_ff=self.d_ff,
            dropout=self.dropout,
            attn=self.attn,
            embed='timeF',
            freq='h',
            activation=self.activation,
            output_attention=False,
            distil=True,
            mix=True,
            device=self.device_
        ).to(self.device_)

        self.model_.load_state_dict(save_dict['model_state_dict'])
        self.model_.eval()

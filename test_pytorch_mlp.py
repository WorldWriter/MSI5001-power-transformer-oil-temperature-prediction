"""
Simple test script to verify PyTorch MLP functionality.
"""

import numpy as np
from models.pytorch_mlp import PyTorchMLPRegressor, get_device


def test_basic_functionality():
    """Test basic fit/predict functionality."""
    print("=" * 60)
    print("Testing PyTorch MLP Basic Functionality")
    print("=" * 60)

    # Check device availability
    print("\n1. Checking device availability...")
    device = get_device()
    print(f"   Selected device: {device}")

    # Generate synthetic data
    print("\n2. Generating synthetic data...")
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    X_train = np.random.randn(n_samples, n_features).astype(np.float32)
    y_train = (X_train[:, 0] * 2 + X_train[:, 1] * 3 +
               np.random.randn(n_samples) * 0.1).astype(np.float32)

    X_test = np.random.randn(200, n_features).astype(np.float32)
    y_test = (X_test[:, 0] * 2 + X_test[:, 1] * 3 +
              np.random.randn(200) * 0.1).astype(np.float32)

    print(f"   Training data shape: {X_train.shape}")
    print(f"   Test data shape: {X_test.shape}")

    # Create and train model
    print("\n3. Creating and training PyTorch MLP...")
    model = PyTorchMLPRegressor(
        hidden_layer_sizes=(64, 32),
        max_iter=50,
        batch_size=32,
        random_state=42,
        early_stopping=True,
        verbose=False,
        device="auto"
    )

    model.fit(X_train, y_train)
    print(f"   Training completed in {model.train_time_:.2f}s")
    print(f"   Number of epochs: {model.n_iter_}")

    # Make predictions
    print("\n4. Making predictions...")
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    r2 = 1 - (np.sum((y_test - y_pred) ** 2) /
              np.sum((y_test - y_test.mean()) ** 2))

    print(f"   MSE: {mse:.4f}")
    print(f"   MAE: {mae:.4f}")
    print(f"   R²: {r2:.4f}")

    # Test with different configurations
    print("\n5. Testing with early stopping disabled...")
    model_no_es = PyTorchMLPRegressor(
        hidden_layer_sizes=(128, 64),
        max_iter=30,
        batch_size=64,
        random_state=42,
        early_stopping=False,
        verbose=False,
        device="auto"
    )

    model_no_es.fit(X_train, y_train)
    y_pred_no_es = model_no_es.predict(X_test)

    mse_no_es = np.mean((y_test - y_pred_no_es) ** 2)
    print(f"   MSE (no early stopping): {mse_no_es:.4f}")
    print(f"   Training time: {model_no_es.train_time_:.2f}s")

    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        test_basic_functionality()
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)

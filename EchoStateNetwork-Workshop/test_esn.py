import numpy as np

# Import the EchoStateNetwork class (assuming it is saved in `esn.py` in the same directory)
from ESN_class_main import EchoStateNetwork


def test_esn():
    # Define parameters for the Echo State Network
    ESN_params = {
        'input_dim': 10,           # Number of input features
        'nodes': 1000,              # Number of reservoir neurons
        'output_dim': 1,           # Number of output features
        'ridge': 1e-6,             # Regularization parameter for ridge regression
        'leak': 0.3,               # Leaking rate
        'connectivity': 0.1,       # Connectivity of the reservoir
        'input_scaling': 0.5,      # Scaling factor for input weights
        'spectral_radius': 0.9,    # Desired spectral radius of the reservoir
        'seed': 42,                # Random seed for reproducibility
        'train_length': 500,       # Number of training steps
        'prediction_length': 100   # Number of prediction steps
    }

    # Create an instance of EchoStateNetwork
    esn = EchoStateNetwork(ESN_params, verbose=2)

    # Test 1: Initialising reservoir with normal distribution
    print("\nTesting with Normal Distribution:")
    W_res_normal = esn.initialize_reservoir(distribution='normal')
    print(f"Spectral radius after scaling: {np.max(np.abs(np.linalg.eigvals(W_res_normal.toarray()))):.4f}")

    # Test 2: Initialising reservoir with uniform distribution
    print("\nTesting with Uniform Distribution:")
    W_res_uniform = esn.initialize_reservoir(distribution='uniform')
    print(f"Spectral radius after scaling: {np.max(np.abs(np.linalg.eigvals(W_res_uniform.toarray()))):.4f}")


if __name__ == "__main__":
    test_esn()

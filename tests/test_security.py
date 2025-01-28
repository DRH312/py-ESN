# This script provides all the security tests put forward in 'validation_requirements.txt'.
import psutil
import pytest
import numpy as np
from pyEchoStateNetwork.ESN_class import EchoStateNetwork

# Establish a default set of parameters for the ESN. Some may be amended within functions.
default_params = {
    "input_dim": 1,
    "nodes": 500,
    "output_dim": 1,
    "distribution": "uniform",
    "leak": 0.3,
    "connectivity": 0.1,
    "spectral_radius": 1.0,
    "seed": 42,
    "bias": False,
    "enable_feedback": False,
    "input_scaling": None,
    'teacher_scaling': None,
    "noise": 0.00,
}


# 3.1 - The model should raise a ValueError if there is an issue with the shapes supplied.
# Here we test no timesteps, and a dimension mismatch.
def test_invalid_inputs():
    params = default_params.copy()
    params["input_dim"] = 1
    params["output_dim"] = 1
    params["nodes"] = 100

    esn = EchoStateNetwork(params, dtype="float64")
    esn.initialize_reservoir()

    # Invalid input: Too few dimensions
    bad_input_1 = np.array([1, 2, 3])  # Should be 2D (K, timesteps)

    with pytest.raises(ValueError):
        esn.acquire_reservoir_states(bad_input_1, np.array([[1, 2, 3]]), visualized_neurons=5, burn_in=10)

    # Invalid input: Too many dimensions
    bad_input_2 = np.random.rand(1, 10, 10)  # Extra dimension should cause failure

    with pytest.raises(ValueError):
        esn.acquire_reservoir_states(bad_input_2, np.random.rand(1, 10), visualized_neurons=5, burn_in=10)

    print("Invalid input test passed!")


# 3.3 - Large datasets should not completely occupy available memory at runtime. On Pycharm there's typically 1200 MB
# available, so we check that it doesn't go over 1 GB for a dataset of 50000 inputs.
# Note, this is my first time tracking memory usage. God knows if I'm remotely correct.
def test_memory_usage_states():
    params = default_params.copy()
    params["nodes"] = 1000
    params["spectral_radius"] = 1.2

    esn = EchoStateNetwork(params, dtype="float64")
    esn.initialize_reservoir()

    # Generate a long timeseries.
    large_timesteps = 50000  # Not crazy big, but enough to be noticeable
    large_inputs = np.random.rand(1, large_timesteps)
    large_targets = np.random.rand(1, large_timesteps)

    # Measure memory use before execution.
    process = psutil.Process()
    mem_before = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    print(f"Memory before execution: {mem_before:.2f} MB")

    esn.acquire_reservoir_states(large_inputs, large_targets, visualized_neurons=5, burn_in=100)

    # Measure memory following state acquisition.
    mem_after = process.memory_info().rss / (1024 ** 2)  # Convert to MB
    print(f"Memory after execution: {mem_after:.2f} MB")

    mem_used = mem_after - mem_before
    print(f"Memory used by ESN: {mem_used:.2f} MB")

    # Set an upper bound for reasonable memory usage, here we use 1000 MB.
    assert mem_used < 1000, f"Memory usage too high! Used: {mem_used:.2f} MB"

    print(f"✅ Memory test passed! Memory used: {mem_used:.2f} MB")


# 3.4 - The model will stop running if the inputs are too large.
def test_oversized_input_handling():
    params = default_params.copy()
    params["input_dim"] = 1
    params["output_dim"] = 1
    params["nodes"] = 5000  # Large reservoir
    params["spectral_radius"] = 1.2
    params["leak"] = 0.3
    params["enable_feedback"] = False

    esn = EchoStateNetwork(params, dtype="float64")
    esn.initialize_reservoir()

    # Generate a massive dataset (too large for normal operation)
    extreme_timesteps = 10_000_000
    extreme_inputs = np.random.rand(1, extreme_timesteps)
    extreme_targets = np.random.rand(1, extreme_timesteps)

    print("\nRunning oversized input test. Expecting termination due to memory limits...")

    # Expect the ESN to raise a MemoryError or custom ValueError
    with pytest.raises((MemoryError, ValueError)) as exc_info:
        esn.acquire_reservoir_states(extreme_inputs, extreme_targets, visualized_neurons=5, burn_in=100)

    print(f"✅ Oversized input test passed! Model correctly raised: {exc_info.type.__name__}")


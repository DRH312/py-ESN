# This script provides all the performance tests put forward in 'validation_requirements.txt'.
import time
import pytest
import numpy as np

from scipy.sparse import csr_matrix
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


# 2.1 - Utilising a sparse representation for W_res is more efficient for computing reservoir states
# than a dense matrix.
def test_sparse_vs_dense_performance():
    params = default_params.copy()
    params["nodes"] = 500
    params["connectivity"] = 0.1

    # Initialise ESN
    esn = EchoStateNetwork(params, verbosity=0, dtype="float64")
    esn.initialize_reservoir()

    # We perform 50 random operation of W_res @ X_t for both the dense and sparse permutations.
    # We track the time for each variant, over each state, and average this time at the end for comparison.
    num_trials = 50
    x_t_batch = esn.rng.random((params["nodes"], num_trials))

    # Create dense version of the reservoir matrix.
    dense_W_res = esn.W_res.toarray()

    # Lists to store execution times
    dense_times = []
    sparse_times = []

    for i in range(num_trials):
        x_t = x_t_batch[:, i:i+1]

        # Benchmark dense matrix update
        start_dense = time.perf_counter()
        _ = dense_W_res @ x_t  # Perform matrix multiplication
        dense_times.append(time.perf_counter() - start_dense)

        # Benchmark sparse matrix update
        start_sparse = time.perf_counter()
        _ = esn.W_res @ x_t  # Use the sparse matrix
        sparse_times.append(time.perf_counter() - start_sparse)

    # Compute mean runtimes
    avg_dense_time = np.mean(dense_times)
    avg_sparse_time = np.mean(sparse_times)

    # Assert sparse is at least 2x faster than dense on average
    assert avg_sparse_time < avg_dense_time, \
        f"Sparse matrix multiplication is not faster. Dense: {avg_dense_time:.6f}s, Sparse: {avg_sparse_time:.6f}s"

    print(f"Sparse vs. Dense Performance Test Passed! Avg Dense: {avg_dense_time:.6f}s, Avg Sparse: {avg_sparse_time:.6f}s")

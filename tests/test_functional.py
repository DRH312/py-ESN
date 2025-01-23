# This script provides all the functional tests put forward in 'validation_requirements.txt'.
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


# 1.1 - Distribution of elements in randomly initialized weights match their parent distributions.
# This test is carried out once for each of the possible distributions offered for initialisation.
@pytest.mark.parametrize("distribution", ["uniform", "normal"])
def test_input_weights_distribution(distribution: str):

    """
    Note that for these default parameters, we tend to get a std for the normal reservoir outputs that is approximately
    0.15. Since the elements are all strongly adjusted by the spectral radius scaling. So we base these tests around
    that known value.

    :param distribution: Either normal or uniform. (str)
    :return:
    """

    if distribution not in ['uniform', 'normal']:
        raise ValueError(f"Distribution must be 'normal' or 'uniform'.")

    params = default_params.copy()
    params["distribution"] = distribution
    params["enable_feedback"] = True

    # Initialize the network - Generate
    esn = EchoStateNetwork(params, dtype='float32', verbose=0)
    esn.initialize_reservoir()

    # Extract input weights
    in_weights = esn.W_in.flatten()
    res_weights = esn.W_res.data
    fb_weights = esn.W_fb.flatten()

    # Uniform Distribution Checks
    if distribution == "uniform":
        # Input weights
        assert np.all((in_weights >= -0.5) & (in_weights <= 0.5)), \
            "Input weights out of bounds for uniform distribution"
        assert np.isclose(np.mean(in_weights), 0,
                          atol=0.05), f"Input weights mean {np.mean(in_weights)} deviates from 0"

        # Reservoir weights
        assert np.all(
            (res_weights >= -0.5) & (res_weights <= 0.5)), "Reservoir weights out of bounds for uniform distribution"
        assert np.isclose(np.mean(res_weights), 0,
                          atol=0.05), f"Reservoir weights mean {np.mean(res_weights)} deviates from 0"

        # Feedback weights
        assert np.all(
            (fb_weights >= -0.5) & (fb_weights <= 0.5)), "Feedback weights out of bounds for uniform distribution"
        assert np.isclose(np.mean(fb_weights), 0,
                          atol=0.05), f"Feedback weights mean {np.mean(fb_weights)} deviates from 0"

    # Normal Distribution Checks
    elif distribution == "normal":
        # Input weights
        assert np.isclose(np.mean(in_weights), 0,
                          atol=0.05), f"Input weights mean {np.mean(in_weights)} deviates from 0"
        assert np.isclose(np.std(in_weights), 1, atol=0.1), f"Input weights std {np.std(in_weights)} deviates from 1"

        # Reservoir weights
        assert np.isclose(np.mean(res_weights), 0,
                          atol=0.05), f"Reservoir weights mean {np.mean(res_weights)} deviates from 0"
        assert np.isclose(np.std(res_weights), 0.15,
                          atol=0.05), f"Reservoir weights std {np.std(res_weights)} deviates from 1"

        # Feedback weights
        assert np.isclose(np.mean(fb_weights), 0,
                          atol=0.05), f"Feedback weights mean {np.mean(fb_weights)} deviates from 0"
        assert np.isclose(np.std(fb_weights), 1, atol=0.1), f"Feedback weights std {np.std(fb_weights)} deviates from 1"


# 1.2 - The same seed should produce the same results... within reason...
def test_esn_reproducibility():
    # Define a simple sine wave for training and testing
    timesteps = 1000
    train_steps = 800
    test_steps = 200
    t = np.linspace(0, 4 * np.pi, timesteps)
    sine_wave = np.sin(t)

    # Training and testing data
    inputs = sine_wave[:-1].reshape(1, -1)  # Input is the sine wave excluding the last point
    targets = sine_wave[1:].reshape(1, -1)  # Target is the sine wave excluding the first point
    train_inputs, train_targets = inputs[:, :train_steps], targets[:, :train_steps]
    test_inputs = inputs[:, train_steps:]


    # Default parameters for the ESN
    params = {
        "input_dim": 1,
        "nodes": 100,
        "output_dim": 1,
        "distribution": "normal",
        "leak": 0.3,
        "connectivity": 0.1,
        "input_scaling": np.array([[0.5, 0.5]]).reshape(-1, 1),  # Scale inputs by 0.5
        "spectral_radius": 1.25,
        "noise": 0.01,
        "enable_feedback": False,
        "seed": 42,
        "bias": True
    }

    # Initialize two ESNs with the same seed
    esn1 = EchoStateNetwork(params, dtype='float64')
    esn2 = EchoStateNetwork(params, dtype='float64')

    # Initialize reservoirs
    esn1.initialize_reservoir()
    esn2.initialize_reservoir()

    # Assert weights are reproducible
    assert np.allclose(esn1.W_res.toarray(), esn2.W_res.toarray(), atol=1e-2, rtol=1e-3), \
        "Reservoir weights are not reproducible within tolerances"
    assert np.allclose(esn1.W_in, esn2.W_in, atol=1e-2, rtol=1e-3), \
        "Input weights are not reproducible within tolerances"
    if esn1.W_fb is not None:
        assert np.allclose(esn1.W_fb, esn2.W_fb, atol=1e-2, rtol=1e-3), \
            "Feedback weights are not reproducible within tolerances"

    # Acquire reservoir states
    esn1.acquire_reservoir_states(train_inputs, train_targets, visualized_neurons=5, burn_in=50)
    esn2.acquire_reservoir_states(train_inputs, train_targets, visualized_neurons=5, burn_in=50)

    # Assert reservoir states and training matrices are consistent
    assert np.allclose(esn1.XX_T, esn2.XX_T, atol=1e-2, rtol=1e-3), \
        "Reservoir state XX_T is not reproducible within tolerances"
    assert np.allclose(esn1.YX_T, esn2.YX_T, atol=1e-2, rtol=1e-3), \
        "Reservoir-target product YX_T is not reproducible within tolerances"

    # Perform ridge regression
    ridge = 1e-8
    W_out1 = esn1.tikhonov_regression(ridge)
    W_out2 = esn2.tikhonov_regression(ridge)

    # Assert readout weights are consistent
    assert np.allclose(W_out1, W_out2, atol=1e-2, rtol=1e-3), \
        "Readout weights are not reproducible within tolerances"

    # Test forecasting
    forecast1, _ = esn1.generative_forecast(T1=test_steps, u1=test_inputs[:, :1], continuation=True)
    forecast2, _ = esn2.generative_forecast(T1=test_steps, u1=test_inputs[:, :1], continuation=True)

    # Assert forecasts are consistent
    assert np.allclose(forecast1, forecast2, atol=1e-2, rtol=1e-3), \
        "Forecasts are not reproducible within tolerances"

    print("Reproducibility test passed successfully!")


# 1.3 - The spectral radius of the reservoir should match user specification.
def test_spectral_radius_scaling():
    params = default_params.copy()
    params["spectral_radius"] = 1.25  # Desired spectral radius
    esn = EchoStateNetwork(params, dtype="float64")
    esn.initialize_reservoir()

    # Convert W_res to dense for eigenvalue calculation
    W_res_dense = esn.W_res.toarray()

    # Calculate the spectral radius
    eigenvalues = np.linalg.eigvals(W_res_dense)
    calculated_radius = np.max(np.abs(eigenvalues))

    # Assert that the spectral radius matches the desired value within tolerance
    assert np.isclose(calculated_radius, params["spectral_radius"], atol=1e-3), \
        f"Spectral radius does not match. Observed: {calculated_radius}, Expected: {params['spectral_radius']}"

    print("Spectral radius scaling test passed!")


# 1.4 - The sparsity of the reservoir should match user specification.
def test_reservoir_sparsity():
    params = default_params.copy()
    params["connectivity"] = 0.1  # Desired sparsity level
    esn = EchoStateNetwork(params, dtype="float32")
    esn.initialize_reservoir()

    # Extract reservoir weights as a dense array
    W_res_dense = esn.W_res.toarray()

    # Calculate sparsity
    num_elements = W_res_dense.size
    num_nonzero = np.count_nonzero(W_res_dense)
    observed_connectivity = num_nonzero / num_elements

    # Assert sparsity matches user specification within tolerance
    assert np.isclose(observed_connectivity, params["connectivity"], atol=1e-2), \
        f"Reservoir connectivity does not match. Observed: {observed_connectivity}, Expected: {params['connectivity']}"

    print("Reservoir sparsity test passed!")


# TODO - Fix this test.
# Attempting to test that the reservoir state update equation is consistent with the literature.
# def test_state_update_consistency():
#     """
#     Proving that the algorithm exactly matches specifications in the literature is practically impossible, since we
#     cannot check every possible outcome for all possible states. But, by manually computing a process of reservoir
#     states, they can be compared to the class method for developing them to check that they are still consistent after
#     any updates are made to the method.
#
#     :return: None
#     """
#
#     # Parameter adjustment.
#     params = default_params.copy()
#     params["enable_feedback"] = True
#     params["nodes"] = 100
#     params["input_dim"] = 1
#     params["output_dim"] = 1
#
#     # Initialise the ESN.
#     esn = EchoStateNetwork(params, dtype="float64", verbose=0)
#     esn.initialize_reservoir()
#
#     # We generate a random series
#     timesteps = 10
#     input_signal = esn.rng.random((params["input_dim"], timesteps), dtype=esn.dtype)
#     feedback_signal = esn.rng.random((params["output_dim"], timesteps), dtype=esn.dtype)
#
#     # Initialize reservoir state
#     manual_states = np.zeros(shape=(params['nodes'], timesteps), dtype=esn.dtype)
#
#     # Manually compute the next reservoir states.
#     for t in range(1, timesteps):
#         # Compute reservoir input contributions
#         u_t = input_signal[:, t:t+1]
#         y_t = feedback_signal[:, t-1:t]
#         x_t = manual_states[:, t-1:t]
#         nonlinear_contribution = (esn.W_res @ x_t + esn.W_in @ u_t + esn.W_fb @ y_t)
#
#         # Update reservoir state
#         manual_states[:, t:t+1] = (1 - params["leak"]) * x_t + params["leak"] * np.tanh(nonlinear_contribution)
#
#     print(manual_states.shape)
#
#     # Use the ESN to compute reservoir states
#     esn_computed_states = esn.acquire_reservoir_states(inputs=input_signal,
#                                                        teachers=feedback_signal,
#                                                        burn_in=0,
#                                                        visualized_neurons=0)
#     print(esn_computed_states.shape)
#
#     print(manual_states[:10, 1])
#     print(esn_computed_states[:10, 1])
#
#     # Assert that states match within tolerance
#     assert np.allclose(manual_states, esn_computed_states, atol=1e-3, rtol=1e-3), \
#         "Reservoir states do not match between manual computation and the ESN's class method."
#
#     print("State update consistency test passed!")


# 1.6 - The shapes of all dot products present in the state-updating equation are all (N, 1).
def test_dot_product_shapes():
    params = default_params.copy()
    params["input_dim"] = 5  # Input size
    params["output_dim"] = 3  # Feedback size
    params["enable_feedback"] = True

    # We only need to generate the weights and create some arbitrary vectors that match the network dimensions.
    esn = EchoStateNetwork(params, verbose=0, dtype="float64")
    esn.initialize_reservoir()

    # Generate some dummy data, it doesn't matter what their actual values are.
    x_t = np.zeros((params["nodes"], 1), dtype=esn.dtype)  # Initial reservoir state
    u_t = esn.rng.random((params["input_dim"], 1), dtype=esn.dtype)  # Input vector
    y_t = esn.rng.random((params["output_dim"], 1), dtype=esn.dtype)

    # Perform dot products of state-update constituents. These are the operations used in the actual model,
    # so long as they work here, they will behave the same effect during operation.
    reservoir_dot = esn.W_res @ x_t
    input_dot = esn.W_in @ u_t
    feedback_dot = esn.W_fb @ y_t if esn.enable_feedback else np.zeros_like(reservoir_dot)

    # Assert shapes
    assert reservoir_dot.shape == (params["nodes"], 1), \
        f"Reservoir dot product shape mismatch: {reservoir_dot.shape}"
    assert input_dot.shape == (params["nodes"], 1), \
        f"Input dot product shape mismatch: {input_dot.shape}"
    if esn.enable_feedback:
        assert feedback_dot.shape == (params["nodes"], 1), \
            f"Feedback dot product shape mismatch: {feedback_dot.shape}"

    # To be safe though we will also perform a single reservoir update with the dummy data, just to check that the class
    # methods output the correct shapes.
    updated_state1 = esn._update_with_feedback(prev_state=x_t, input_pattern=u_t, target=y_t)
    assert updated_state1.shape == (params["nodes"], 1), \
        f"Reservoir update produces incorrect shape: {updated_state1.shape}. Should be ({params['nodes']}, 1)."
    updated_state2 = esn._update_no_feedback(prev_state=x_t, input_pattern=u_t)
    assert updated_state2.shape == (params["nodes"], 1), \
        f"Reservoir update produces incorrect shape: {updated_state2.shape}. Should be ({params['nodes']}, 1)."

    print("All dot product shapes are correct!")

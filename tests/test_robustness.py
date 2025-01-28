# This script provides all the robustness tests put forward in 'validation_requirements.txt'.
import pytest
import random
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


# 2.1) The internal states should not diverge when hyperparameters are within reasonable bounds.
def test_internal_state_stability():
    """
    We use random search over parameter space to assure reasonable behaviour instead of doing an
    exhaustive search, as it would take too long.
    :return:
    """

    # Define reasonable hyperparameter ranges
    spectral_radii = [0.5, 0.9, 1.2, 1.5]  # <1.5 to avoid chaos
    leak_rates = [0.1, 0.3, 0.5, 0.7]  # Common leak values
    input_scalings = [0.1, 0.5, 1.0, 2.0]  # Should not overwhelm reservoir

    num_trials = 30  # Number of random selections
    random.seed(42)  # Ensure repeatability
    failed_cases = []

    # Randomly select hyperparameter combinations
    for _ in range(num_trials):
        params = default_params.copy()
        params["input_dim"] = 1
        params["nodes"] = 200  # By necessity, we keep this small.
        params["spectral_radius"] = random.choice(spectral_radii)
        params["leak"] = random.choice(leak_rates)
        params["input_scaling"] = np.array([random.choice(input_scalings)]).reshape(-1, 1)
        params["enable_feedback"] = False

        # Initialise ESN
        esn = EchoStateNetwork(params, verbosity=0, dtype="float64")
        esn.initialize_reservoir()

        # Generate sine wave for testing
        timesteps = 500
        t = np.linspace(0, 4 * np.pi, timesteps)
        sine_wave = np.sin(t).reshape(1, -1)

        # Acquire reservoir states
        reservoir_states = esn.acquire_reservoir_states(sine_wave, sine_wave, visualized_neurons=5, burn_in=50)

        # Compute maximum absolute state per timestep
        max_state_per_timestep = np.max(np.abs(reservoir_states), axis=0)  # Shape: [T]

        # Compute differences (gradients) between successive reservoir states
        state_differences = np.diff(reservoir_states, axis=1)  # Shape: [N, T-1]
        max_state_diff = np.max(np.abs(state_differences))  # Largest change across all neurons

        # Check if reservoir states diverge
        if np.max(max_state_per_timestep) > 10 or max_state_diff > 5:
            failed_cases.append((params["spectral_radius"], params["leak"], params["input_scaling"],
                                 np.max(max_state_per_timestep), max_state_diff))

    # Final assertion - There should be no failed cases within this hyperparameter space.
    # If any permutations failed, it will output the set that led to failure, and the erroneous gradient.
    assert len(failed_cases) == 0, \
        f"Reservoir states diverged in {len(failed_cases)} cases:\n" + \
        "\n".join([f"SR={case[0]}, Leak={case[1]}, Scale={case[2]}, MaxState={case[3]:.3f}, MaxGrad={case[4]:.3f}"
                   for case in failed_cases])

    print("Internal state stability test passed across randomized hyperparameter selections.")


# 4.2 - The model should be fairly tolerant to noise.
# We test on a basic sine wave prediction.
def test_forecasting_robustness_to_noise():
    params = default_params.copy()
    params["spectral_radius"] = 1.2
    params["noise"] = 0.0  # Start with no noise for the baseline.

    # Generate sine wave data for training & testing
    timesteps = 1000
    train_steps = 800
    test_steps = 200
    t = np.linspace(0, 4 * np.pi, timesteps)
    sine_wave = np.sin(t)

    # Training and testing data
    inputs = sine_wave[:-1].reshape(1, -1)
    targets = sine_wave[1:].reshape(1, -1)
    train_inputs, train_targets = inputs[:, :train_steps], targets[:, :train_steps]
    test_inputs, test_targets = inputs[:, train_steps:], targets[:, train_steps:]

    # Train first ESN (clean data)
    esn_clean = EchoStateNetwork(params, verbosity=0, dtype="float64")
    esn_clean.initialize_reservoir()
    esn_clean.acquire_reservoir_states(train_inputs, train_targets, visualized_neurons=5, burn_in=50)
    esn_clean.W_out = esn_clean.tikhonov_regression(ridge=1e-6)
    forecast_clean, _ = esn_clean.generative_forecast(T1=test_steps, u1=test_inputs[:, :1], continuation=True)

    # Train second ESN (noisy data)
    params_noisy = params.copy()
    params_noisy["noise"] = 0.05  # Inject noise directly into the ESN
    esn_noisy = EchoStateNetwork(params_noisy, verbosity=0, dtype="float64")
    esn_noisy.initialize_reservoir()
    esn_noisy.acquire_reservoir_states(train_inputs, train_targets, visualized_neurons=5, burn_in=50)
    esn_noisy.W_out = esn_noisy.tikhonov_regression(ridge=1e-6)
    forecast_noisy, _ = esn_noisy.generative_forecast(T1=test_steps, u1=test_inputs[:, :1], continuation=True)

    # Compute RMSE for both forecasts
    rmse_clean = np.sqrt(np.mean((forecast_clean[:, 1:] - test_targets) ** 2))
    rmse_noisy = np.sqrt(np.mean((forecast_noisy[:, 1:] - test_targets) ** 2))

    # Allowable degradation threshold
    degradation_threshold = 5  # RMSE should not increase fivefold for a subtle increase in noise.

    # Assert that the RMSE does not degrade excessively due to noise
    assert rmse_noisy < rmse_clean * degradation_threshold, \
        (f"Forecasting performance degraded too much due to noise! Clean RMSE: "
         f"{rmse_clean:.4f}, Noisy RMSE: {rmse_noisy:.4f}")

    print(f"Forecasting robustness test passed! Clean RMSE: {rmse_clean:.4f}, Noisy RMSE: {rmse_noisy:.4f}")


# 4.3 - Testing the model's resilience to missing datapoints.
def test_missing_values_state_consistency():
    params = default_params.copy()
    params["spectral_radius"] = 1.2

    # Generating sine wave data.
    timesteps = 1000
    train_steps = 800
    t = np.linspace(0, 4 * np.pi, timesteps)
    sine_wave = np.sin(t)

    # Prepare input and target sequences.
    inputs = sine_wave[:-1].reshape(1, -1)  # Input: sine wave (shifted one step back)
    targets = sine_wave[1:].reshape(1, -1)  # Target: sine wave (original)
    train_inputs, train_targets = inputs[:, :train_steps], targets[:, :train_steps]

    # Run ESN on pure data.
    esn_clean = EchoStateNetwork(params, verbosity=0, dtype="float64")
    esn_clean.initialize_reservoir()
    states_clean = esn_clean.acquire_reservoir_states(train_inputs, train_targets, visualized_neurons=5, burn_in=50)

    # Introduce NaNs into training data.
    nan_mask = esn_clean.rng.choice([True, False], size=train_inputs.shape, p=[0.05, 0.95])  # 5% NaNs
    train_inputs_missing = train_inputs.copy()
    train_inputs_missing[nan_mask] = np.nan

    # Run ESN on missing-data input
    esn_missing = EchoStateNetwork(params, dtype="float64")
    esn_missing.initialize_reservoir()
    states_missing = esn_missing.acquire_reservoir_states(train_inputs_missing,
                                                          train_targets,
                                                          visualized_neurons=5, burn_in=50)

    # Calculate the mean deviation between state matrices.
    mean_deviation = np.mean(np.abs(states_clean - states_missing))

    # Assert that the deviation is below an arbitrary threshold.
    assert mean_deviation < 0.1, \
        f"Reservoir states diverged too much with missing data! Mean deviation: {mean_deviation:.4f}"

    print("Missing values test passed! ESN remains consistent despite missing data.")

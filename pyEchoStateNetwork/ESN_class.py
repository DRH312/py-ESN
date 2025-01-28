# Libraries necessary for core functionality:
import numpy as np
from scipy import linalg, sparse

# Libraries required for packaging and writing data to local directories:
import os
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Brevity
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")


"""
This Python file contains the primary classes that constitute a portable echo-state network!
Some functionality is kept in separate files, including loss functions and random-weight 
generation. The aim is to provide a simple but customisable ESN that can be imported into various
use-cases. 

"""


def esn_params_guide():
    """
    Prints the required parameters for initializing an Echo State Network.
    """
    params = {
        "input_dim": "Number of input features (K).",
        "nodes": "Number of reservoir neurons (N).",
        "output_dim": "Number of output features (L).",
        "distribution": "Weight initialization distribution ('uniform' or 'normal').",
        "leak": "Leak rate controlling state update speed.",
        "connectivity": "Fraction of nonzero weights in the reservoir.",
        "input_scaling": "Scaling factor applied to input weights.",
        "spectral_radius": "Spectral radius of the reservoir matrix.",
        "noise": "Amount of noise added to the reservoir state.",
        "enable_feedback": "Boolean flag to enable output-to-reservoir feedback.",
        "seed": "Random seed for reproducibility.",
        "bias": "Boolean flag to include a bias term."
    }

    print("\n **Required Parameters for ESN:**")
    for key, desc in params.items():
        print(f" - **{key}**: {desc}")


class EchoStateNetwork:

    def __init__(self, ESN_params: dict, dtype: str = "float64", verbosity=0):

        """
        Declaring this class fully initializes an Echo State Network provided that a correct dictionary of inputs is
        supplied to the class.

        :param ESN_params: A dictionary of model parameters. Call param_guide function for more information.
        :param dtype: Currently acceptable datapoints are float32 and float64.
        :param verbosity: An integer ranging from 0 to 3. Increasingly provides system information at runtime.
        """

        # Model structure:
        self.K = ESN_params['input_dim']  # The number of features in the input vector.
        self.N = ESN_params['nodes']  # The number of neurons in the reservoir.
        self.L = ESN_params['output_dim']  # The number of features in the output vector.
        self.distribution = ESN_params['distribution']

        # Model hyperparameters:
        self.leak = ESN_params['leak']
        self.connectivity = ESN_params['connectivity']
        self.input_scaling = ESN_params['input_scaling']
        self.sr = ESN_params['spectral_radius']
        self.noise = ESN_params['noise']

        # Parameters that affect the structure of the network.
        self.enable_feedback = ESN_params['enable_feedback']
        if self.enable_feedback:
            self.teacher_scaling = ESN_params['teacher_scaling']
        else:
            self.teacher_scaling = None

        # Determines random number generation.
        self.seed = int(ESN_params['seed'])
        self.rng = np.random.default_rng(self.seed)

        # Whether bias is incorporated into the input and output weights.
        self.bias = ESN_params['bias']

        # Cementing the datatype which will be maintained across the whole series of computations.
        self.dtype = dtype
        if self.dtype not in ["float64", "float32"]:
            raise ValueError(f"Selected datatype must adhere to {['float64', 'float32']}")

        # Ensuring that the verbosity scale is an integer, and that it falls into the acceptable range.
        self.verbosity = int(verbosity)
        if self.verbosity > 3:
            self.verbosity = 3

        # Model weights:
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.W_fb = None

        # Matrices relevant for training readout. They will be properly initialised during state acquisition.
        self.XX_T = None  # The square matrix of the reservoir outputs.
        self.YX_T = None  # The matrix product of the target outputs with the transpose of the concatenated vector.

        # Later these will be required for producing extra reservoir states or to forecast system evolutions.
        self.last_state = None
        self.last_input = None
        self.last_output = None



    def load_network(self, Win: np.ndarray, Wres: np.ndarray, Wfb: np.ndarray = None):
        """
        Allows you to upload a pre-existing model so long as the dimensions of the inputted matrices match the ESN
        parameter dictionary.

        :param Win: Must have shape (N, K + bias).
        :param Wres: Must have shape (N, N).
        :param Wfb: Optional.Must have shape (N, L). Only necessary if feedback is enabled.

        :return: None.
        """

        # Check Win shape.
        expected_Win_shape = (self.N, self.K + self.bias)
        if Win.shape != expected_Win_shape:
            raise ValueError(f"Win shape mismatch. Expected {expected_Win_shape}, but received {Win.shape}.")

        # Check Wres shape.
        expected_Wres_shape = (self.N, self.N)
        if Wres.shape != expected_Wres_shape:
            raise ValueError(f"Wres shape mismatch. Expected {expected_Wres_shape}, but received {Wres.shape}.")

        # Check Wfb shape (if feedback is enabled).
        if self.enable_feedback:
            expected_Wfb_shape = (self.N, self.L)
            if Wfb is None:
                raise ValueError(
                    f"Feedback is enabled, but no Wfb matrix was provided! Expected shape: {expected_Wfb_shape}.")
            if Wfb.shape != expected_Wfb_shape:
                raise ValueError(f"Wfb shape mismatch. Expected {expected_Wfb_shape}, but received {Wfb.shape}.")
        else:
            Wfb = None  # Explicitly set to None if feedback is disabled.

        # Assign new weights, asserting the correct datatype.
        self.W_in = Win.astype(self.dtype)
        self.W_res = sparse.csr_matrix(Wres.astype(self.dtype))  # Convert to sparse matrix.
        self.W_fb = Wfb.astype(self.dtype) if Wfb is not None else None

        print("Network weights uploaded.")


    def initialize_reservoir(self, save_weights=False) -> None:

        """
        Initializes an N*N sparse adjacency matrix that defines the reservoir of the ESN. The nonzero elements of the
        matrix can be sampled from either a uniform or standard normal distribution.

        :return: csr_matrix: Sparse reservoir matrix with the desired spectral radius.
        """

        # Check that the reservoir distribution requested is a valid input.
        if self.distribution not in ["uniform", "normal"]:
            raise ValueError(f"Selected distribution must be 'uniform' or 'normal'")

        # Generate sparsity mask. This should be reproducible amongst consistent seeds.
        mask = self.rng.random((self.N, self.N)) < self.connectivity
        num_nonzeros = mask.sum()


        if self.distribution == 'normal':
            # Sample reservoir weights from a standard Gaussian distribution.

            # Initializing the reservoir adjacency matrix.
            # Generates exactly as many nonzero elements as are defined by the mask.
            self.W_res = self.rng.normal(loc=0, scale=1, size=num_nonzeros).astype(self.dtype)
            # Rescale to ensure standard deviation is 1
            self.W_res /= np.std(self.W_res)

            # Initialising the input connection weights.
            # These are sampled from the same distribution as the reservoir, but remain dense.
            self.W_in = self.rng.normal(loc=0, scale=1, size=(self.N, self.K + self.bias)).astype(self.dtype)
            # Setting the range to [-1, +1]

            # If feedback is enabled, generated a feedback matrix.
            if self.enable_feedback:
                self.W_fb = self.rng.normal(loc=0, scale=1, size=(self.N, self.L)).astype(self.dtype)

            else:
                self.W_fb = None


        elif self.distribution == 'uniform':
            # Sample reservoir weights from a symmetric uniform distribution.

            # Initializing the reservoir adjacency matrix.
            self.W_res = self.rng.uniform(low=-0.5, high=0.5, size=num_nonzeros).astype(self.dtype)
            # Rescale to ensure range is consistent
            self.W_res /= np.abs(self.W_res).max()  # Scale to [-0.5, 0.5]

            # Initializing the input connection weights.
            # These are sampled from the same distribution as the reservoir, but remain dense.
            self.W_in = self.rng.uniform(low=-0.5, high=0.5, size=(self.N, self.K + self.bias)).astype(self.dtype)

            # If feedback is enabled, generated a feedback matrix.
            if self.enable_feedback:
                self.W_fb = self.rng.uniform(low=-0.5, high=0.5, size=(self.N, self.L)).astype(self.dtype)

            else:
                self.W_fb = None


        # We apply scaling onto the weight matrices instead of the inputs:
        if self.input_scaling is not None:
            if self.input_scaling.shape != (self.K + self.bias, 1):
                raise ValueError(f"The scaling vector should have shape ({self.K + self.bias}, 1).")
            self.W_in = self.W_in * self.input_scaling.T

        if self.teacher_scaling is not None:
            if self.teacher_scaling.shape != (self.L, 1):
                raise ValueError(f"The scaling vector should have shape ({self.L}, 1).")
            self.W_fb = self.W_fb * self.teacher_scaling.T

        # Post scaling, we assert that the first column of W_in consists of all 1s.
        if self.bias:
            self.W_in[:, 0] = 1

        # Convert W_res into a sparse matrix to speed up state acquisition later.
        row_indices, col_indices = np.where(mask)
        self.W_res = sparse.csr_matrix((self.W_res, (row_indices, col_indices)),
                                       shape=(self.N, self.N), dtype=self.dtype)


        if self.verbosity > 0:
            print("Reservoir adjacency matrix initialized. Beginning spectral radius scaling.")

        # Now that matrices are generated, scale the spectral radius of the reservoir.
        self._scale_spectral_radius()

        if self.verbosity > 0:
            print("Reservoir weights spectral radius scaling completed.")

        if self.verbosity > 1:
            self._plot_reservoir_histogram()

            # Print a table of matrix shapes
            print("\n=== Matrix Shapes ===")
            print(f"{'Matrix':<15}{'Shape':<20}")
            print(f"{'-' * 35}")
            print(f"{'W_res':<15}{str(self.W_res.shape):<20}")
            print(f"{'W_in':<15}{str(self.W_in.shape):<20}")
            if self.W_fb is not None:
                print(f"{'W_fb':<15}{str(self.W_fb.shape):<20}")
            else:
                print(f"{'W_fb':<15}{'None':<20}")

        if save_weights:
            self._save_weights_locally()


    def _save_weights_locally(self) -> None:
        if self.verbosity > 1:
            print(f"Reservoir Adjacency Matrix is of type {type(self.W_res)} with shape {self.W_res.shape}")

        # Defining the output directory relative to the package.
        output_dir = os.path.join(os.getcwd(), "generated_weights")
        os.makedirs(output_dir, exist_ok=True)


        # Save reservoir weights.
        W_res_dense = self.W_res.toarray()  # Converting to dense for uploading to CSV.
        pd.DataFrame(W_res_dense).to_csv(os.path.join(output_dir, f"W_res-{timestamp}.csv"), index=False,
                                         header=False)

        # Save W_in
        pd.DataFrame(self.W_in).to_csv(os.path.join(output_dir, f"W_in-{timestamp}.csv"), index=False,
                                       header=False)

        # Save W_fb if feedback is enabled
        if self.W_fb is not None:
            pd.DataFrame(self.W_fb).to_csv(os.path.join(output_dir, f"W_fb-{timestamp}.csv"), index=False,
                                           header=False)

        print(f"Network matrices uploaded to {output_dir}")


    def _scale_spectral_radius(self) -> None:

        """
        Scales the spectral radius of the reservoir matrix to match user-specification.
        """

        largest_eigenvalue = np.abs(sparse.linalg.eigs(self.W_res, k=1, which='LM',
                                                       return_eigenvectors=False,
                                                       tol=1e-10)[0])

        # Scale the sparse reservoir matrix.
        scale_factor = self.sr / largest_eigenvalue
        self.W_res *= scale_factor

        if self.verbosity > 0:
            new_sr = np.abs(sparse.linalg.eigs(self.W_res, k=1, which='LM',
                                               return_eigenvectors=False,
                                               tol=1e-10)[0])
            print(f"Reservoir spectral radius scaled to: {new_sr}")


    def _plot_reservoir_histogram(self) -> None:

        """
        Plots a histogram of the reservoir matrix elements to assure the user of correct implementation.
        """

        # Checking that there is actually data to plot.
        if len(self.W_res.data) == 0:
            raise ValueError("There are no non-zero values in the reservoir matrix to plot.")

        elif len(self.W_in.flatten()) == 0:
            raise ValueError("There are no non-zero values in the input connection matrix to plot.")

        else:
            num_plots = 3 if self.enable_feedback else 2
            fig, axes = plt.subplots(1, num_plots, figsize=(15, 5))

            # Plot histogram for the input weights
            axes[0].hist(self.W_in.flatten(), bins=50, density=True, color='orange', edgecolor='black', alpha=0.75)
            axes[0].set_title("Input Weights Distribution")
            axes[0].set_xlabel("Value")
            axes[0].set_ylabel("Density")
            axes[0].grid(True)

            # Plot histogram for the reservoir weights
            axes[1].hist(self.W_res.data, bins=50, density=True, color='blue', edgecolor='black', alpha=0.75)
            axes[1].set_title("Reservoir Weights Distribution")
            axes[1].set_xlabel("Value")
            axes[1].set_ylabel("Density")
            axes[1].grid(True)

            # Plot histogram for feedback weights if feedback is enabled.
            if self.enable_feedback:
                # Check that there is feedback data to plot.
                if len(self.W_fb.flatten()) == 0:
                    raise ValueError("There are no non-zero values in the feedback connection matrix to plot.")

                axes[2].hist(self.W_fb.flatten(), bins=50, density=True, color='green', edgecolor='black', alpha=0.75)
                axes[2].set_title("Feedback Weights Distribution")
                axes[2].set_xlabel("Value")
                axes[2].set_ylabel("Density")
                axes[2].grid(True)

            # Adjust layout and display the plot
            plt.tight_layout()
            plt.show()


    def _update_no_feedback(self, prev_state, input_pattern) -> np.ndarray:
        """
        Performs a singular reservoir update. No feedback is utilised here.

        :param prev_state: The preceding reservoir state, with shape (N, 1).
        :param input_pattern: The current input vector, with shape (K, 1).

        :return: The current reservoir state, with shape (N, 1) as well.
        """

        if input_pattern.ndim == 1:
            input_pattern = input_pattern.reshape(-1, 1)
        if prev_state.ndim == 1:
            prev_state = prev_state.reshape(-1, 1)

        # The matrix products for the inputs and previous reservoir sates.
        nonlinear_contribution = self.W_in @ input_pattern + self.W_res @ prev_state
        nonlinear_contribution = nonlinear_contribution.reshape(-1, 1)

        # Generate reproducible noise with a Gaussian distribution.
        random_noise = self.rng.normal(loc=0, scale=1, size=(self.N, 1)) * self.noise

        # Compute the pre-activation, including the noise term.
        preactivation = np.tanh(nonlinear_contribution) + random_noise


        return (1 - self.leak) * prev_state + self.leak * preactivation


    def _update_with_feedback(self, prev_state, input_pattern, target) -> np.ndarray:

        """
        Performs a singular reservoir update, including contributions from feedback.

        :param prev_state: The preceding reservoir state, with shape (N, 1)
        :param input_pattern: The current input vector, with shape (K, 1).
        :param target: The current target vector, with shape (L, 1).

        :return: The current reservoir state, with shape (N, 1) as well.
        """

        if input_pattern.ndim == 1:
            input_pattern = input_pattern.reshape(-1, 1)
        if prev_state.ndim == 1:
            prev_state = prev_state.reshape(-1, 1)
        if target.ndim == 1:
            target = target.reshape(-1, 1)

        # The matrix products for the inputs, previous reservoir states and feedback.
        nonlinear_contribution = self.W_in @ input_pattern + self.W_res @ prev_state + self.W_fb @ target
        nonlinear_contribution = nonlinear_contribution.reshape(-1, 1)

        # Generate reproducible noise with a Gaussian distribution.
        random_noise = self.rng.normal(loc=0, scale=1, size=(self.N, 1)) * self.noise

        # Compute the pre-activation, including the noise term.
        preactivation = np.tanh(nonlinear_contribution) + random_noise

        return (1 - self.leak) * prev_state + preactivation


    def acquire_reservoir_states(self,
                                 inputs: np.ndarray,
                                 teachers: np.ndarray,
                                 visualized_neurons: int,
                                 burn_in: int,
                                 NaNdling: int = 0) -> np.ndarray:

        """
        Perform dimensionality expansion on the data used to train the network. Conventionally, this will usually be
        the past of a signal, for which forecasting is used to predict the signal's evolution. #

        :param burn_in: The number of initial reservoir states to discard as they misrepresent the data.
        :param inputs: Input sequence with shape (K, timesteps).
        :param teachers: Target sequence with shape (L, timesteps). Not optional. Required for regression.
        :param visualized_neurons: The number of neurons to be plotted. This parameter is only necessary if verbosity is
        greater than 1.
        :param NaNdling: 0 -> replace w/ zeros. 1 -> interpolate.

        :return: Reservoir states with shape (N, timesteps).
        """

        if inputs.shape[0] != self.K or inputs.ndim != 2:
            raise ValueError(f"Input signal should have shape ({self.K}, timesteps). Has shape: {inputs.shape}")

        if inputs.shape[1] > 1e6:
            raise MemoryError(f"I admire your optimism, but this is far too many timesteps. <1 million please.")

        if teachers is None:
            raise ValueError("Training targets (teachers) must be provided regardless of whether feedback is enabled.")

        # Validate that inputs and teachers have the same number of timesteps.
        if teachers is not None and inputs.shape[1] != teachers.shape[1]:
            raise ValueError(
                f"Mismatch in timesteps: Inputs have {inputs.shape[1]} timesteps, "
                f"while teachers have {teachers.shape[1]} timesteps."
            )

        if np.isnan(inputs).any():

            # Set NaNs to 0.
            if NaNdling == 0:
                print("Warning: NaN values detected in inputs. Setting missing values to zero.")
                inputs = np.nan_to_num(inputs, nan=0.0)

            # Interpolate
            if NaNdling == 1:
                print("Warning: NaN values detected in inputs. Applying interpolation to handle missing data.")
                for i in range(inputs.shape[0]):  # Iterate over input dimensions (each feature separately)
                    nan_mask = np.isnan(inputs[i, :])  # Identify where NaNs occur along this row.
                    if np.any(nan_mask):
                        valid_timesteps = np.where(~nan_mask)[0]  # Locating timesteps without NaNs.
                        inputs[i, nan_mask] = np.interp(x=np.where(nan_mask)[0],
                                                        xp=valid_timesteps,
                                                        fp=inputs[i, valid_timesteps])

        inputs = inputs.astype(self.dtype)

        if self.bias:
            inputs = np.vstack([np.ones((1, inputs.shape[1]), dtype=self.dtype), inputs])

        # Validate the number of neurons to visualize.
        if visualized_neurons > self.N:
            raise ValueError(
                f"visualized_neurons ({visualized_neurons}) cannot exceed the number of reservoir neurons ({self.N}).")

        # Allocate memory for the matrices utilised in training.
        full_state_dim = int(self.bias) + self.K + self.N
        self.XX_T = np.zeros(shape=(full_state_dim, full_state_dim), dtype=self.dtype)
        self.YX_T = np.zeros(shape=(self.L, full_state_dim), dtype=self.dtype)

        # Initialize the base reservoir state, and determine the "length" of the signal.
        timesteps = inputs.shape[1]
        states = np.zeros(shape=(self.N, timesteps), dtype=self.dtype)

        # Iterate over the timesteps and perform dimensionality expansion to generate reservoir states.
        # With feedback:
        if self.enable_feedback:
            for t in range(1, timesteps):
                states[:, t:t+1] = self._update_with_feedback(prev_state=states[:, t - 1:t],
                                                              input_pattern=inputs[:, t:t+1],
                                                              target=teachers[:, t-1:t]).astype(self.dtype)

                # Create augmented state vector [1; u[t]; x[t]] for this timestep.
                augmented_state = np.vstack(tup=[inputs[:, t:t+1], states[:, t:t+1]])

                # Update XX^T and YX^T matrices incrementally.
                if t >= burn_in:
                    self.XX_T += augmented_state @ augmented_state.T
                    self.YX_T += teachers[:, t:t+1] @ augmented_state.T

        # Without feedback:
        else:
            for t in range(1, timesteps):
                states[:, t:t+1] = self._update_no_feedback(prev_state=states[:, t-1:t],
                                                            input_pattern=inputs[:, t:t+1]).astype(self.dtype)

                # Create augmented state vector [1; u[t]; x[t]] for this timestep.
                augmented_state = np.vstack(tup=[inputs[:, t:t+1], states[:, t:t+1]])

                # Update XX^T and YX^T matrices incrementally.
                if t >= burn_in:
                    self.XX_T += augmented_state @ augmented_state.T
                    self.YX_T += teachers[:, t:t+1] @ augmented_state.T


        # Final step, retain the final states for continuation or forecasting later.
        self.last_state = states[:, -1]
        self.last_input = inputs[:, -1]
        self.last_output = teachers[:, -1]


        # Printing a subset of the reservoir activations over time.
        if self.verbosity > 1:
            plt.figure(figsize=(10, 6))
            plt.title(f"Activation of {int(visualized_neurons)} Reservoir Neurons Over Time", fontsize=12)
            plt.ylabel("Node Activation", fontsize=10)
            plt.xlabel("Time Steps", fontsize=10)
            for i in range(visualized_neurons):
                plt.plot(range(timesteps - burn_in), states[i, burn_in:], lw=0.5, label=f"Neuron {i + 1}")
            plt.legend(fontsize=8, loc="upper right", ncol=2, frameon=False)
            plt.tight_layout()
            plt.show()

        if self.verbosity > 1:
            print(f"XX^T has shape: {self.XX_T.shape}")
            print(f"YX^T has shape: {self.YX_T.shape}")

        return states.astype(self.dtype)


    def tikhonov_regression(self, ridge: float) -> np.ndarray:

        """
        Performs Tikhonov (ridge) regression using the reservoir states in a one-step optimisation approach.

        :param ridge: The penalty applied to readout weight matrix to prevent any elements from dominating.

        :return: A readout weight matrix used for further predictions or system-evolution forecasting.
        """

        # Apply a penalty to the diagonal elements of XX^T.
        regularized_XX_T = self.XX_T + (ridge * np.eye(self.XX_T.shape[0], dtype=self.dtype))

        # Using a linear solver that will no doubt be better than anything I can make.
        # We want to ensure that XX^T is symmetric for the following to work.
        self.W_out = linalg.solve(regularized_XX_T, self.YX_T.T, assume_a='sym').T

        if self.verbosity > 0:
            print(f"Readout weight matrix shape: {self.W_out.shape}")

            # If verbosity is high, plot the histogram of the readout weights
        if self.verbosity > 1:
            self._plot_readout_histogram()

            # If verbosity is even higher, save the weights to a file
        if self.verbosity > 2:
            current_dir = os.getcwd()
            parent_dir = os.path.dirname(current_dir)
            output_dir = os.path.join(parent_dir, "Generated_Weights")
            os.makedirs(output_dir, exist_ok=True)

            pd.DataFrame(self.W_out).to_csv(
                os.path.join(output_dir, f"W_out-{timestamp}.csv"), index=False, header=False
            )
            print(f"Readout weights saved to {output_dir}")

        return self.W_out


    def _plot_readout_histogram(self) -> None:
        """
        Plots a histogram of the readout weight matrix elements.
        """
        if self.W_out is None:
            raise ValueError("Readout weight matrix has not been computed yet.")

        plt.figure(figsize=(10, 6))
        plt.hist(self.W_out.flatten(), bins=50, density=True, color='purple', edgecolor='black', alpha=0.75)
        plt.title("Readout Weight Distribution")
        plt.xlabel("Weight Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    def cross_series_prediction(self,
                                input_signal: np.ndarray,
                                initial_state: np.ndarray = None,
                                initial_output: np.ndarray = None,
                                continuation=True) -> tuple:
        """
        When supplying an input signal, do so without including a bias atop each datapoint. This method will
        automatically add it onto the signal if the network utilises it. The input signal should have shape
        (K, timesteps), for which the columns will be used to determine how many forecasting iterations take place.


        :param input_signal: A series of inputs with shape (K, forecast_horizon) used to predict another signal.
        :param initial_state:  The dimensionality expanded input of the input preceding the input_signal. (N, 1).
        :param initial_output: The value of the output at the timestep preceding the initial input. (L, 1).
        :param continuation: Whether to use the last states used during reservoir warmup as part of this exercise.
        :return:
        """


        # ----- VALIDATION -----
        # Perform some checks to ensure everything needed for functionality is present.
        if self.W_out is None:
            raise ValueError("Readout Weights are undefined. The network must be trained before it can forecast.")

        # Input signal dimensions consistent with network, timescale consistent with forecasting horizon.
        if input_signal.shape[0] != self.K:
            raise ValueError(f"Shape mismatch with number of rows for input. Input signal should have shape "
                             f"({self.K + self.bias}, timesteps). Input has shape {input_signal.shape}")
        timesteps = input_signal.shape[1]

        # If the network uses a bias, the inputs here will need to have one too.
        if self.bias:
            input_signal = np.vstack([np.ones((1, timesteps)), input_signal])

        # ----- INITIALIZATION -----
        if self.enable_feedback:  # We add some flak to the beginning of the matrix to use the initial values.
            Y_out = np.zeros(shape=(self.L, 1 + timesteps), dtype=self.dtype)
        else:
            Y_out = np.zeros(shape=(self.L, timesteps), dtype=self.dtype)


        # ----- SETTING UP INITIAL STATES -----
        if continuation:  # Use the final states generated during reservoir state acquisition.
            # Acquiring x_0 = x_T:
            if self.last_state is None:
                raise ValueError(f"Continuation is set to true, but there are no saved states.")
            last_state = self.last_state

            # Acquiring y_0 = y_T:
            if self.enable_feedback:
                if self.last_output is None:
                    raise ValueError(f"Continuation and feedback are set to true, but there is no saved last output.")
                Y_out[:, 0] = self.last_output

        else:
            # Use the provided initial_state if available, else generate it as a null-vector.
            if initial_state is not None and initial_state.shape == (self.N, 1):
                last_state = initial_state
            else:
                last_state = np.zeros((self.N, 1), dtype=self.dtype)

            if self.enable_feedback:
                if initial_output is not None and initial_output.shape == (self.L, 1):
                    Y_out[:, 0] = initial_output
                else:
                    Y_out[:, 0] = np.zeros((self.L, 1), dtype=self.dtype)


        # ----- CROSS SERIES PREDICTION -----
        if self.enable_feedback:
            # Pad out the first column of the input vector with zeros. Neatens notation going forward.
            input_signal = np.hstack([np.zeros((self.K + self.bias, 1)), input_signal])

            # Loop for feedback-enabled predictions.
            for t in range(1, timesteps + 1):
                # Variables needed for computation.
                input_pattern = input_signal[:, t:t+1]  # Current timestep.
                output_feedback = Y_out[:, t-1:t]  # Previous timestep.

                current_state = self._update_with_feedback(prev_state=last_state,
                                                           input_pattern=input_pattern,
                                                           target=output_feedback)

                # Predict the state of the other signal at this timestep.
                Y_out[:, t:t+1] = self.W_out @ np.vstack([input_pattern, current_state])

                # Set this state to be the 'last state' for subsequent use.
                last_state = current_state

            return Y_out[:, 0:], last_state

        else:  # No feedback.
            for t in range(0, timesteps):
                input_pattern = input_signal[:, t:t + 1]
                current_state = self._update_no_feedback(prev_state=last_state,
                                                         input_pattern=input_pattern)

                # Predict the state of the other signal at this timestep.
                Y_out[:, t:t+1] = self.W_out @ np.vstack([input_pattern, current_state])

                # Set this state to be the 'last state' for subsequent use.
                last_state = current_state

            return Y_out[:, :], last_state




    def generative_forecast(self,
                            T1: int,
                            u1: np.ndarray = None,
                            x0: np.ndarray = None,
                            continuation=True) -> tuple:

        """
        Performs generative forecasting, where the predicted output is used as the subsequent input. Do not apply a bias
        to the input even if bias is enabled for the network. The method will pick up on this automatically and apply
        it if needed. Feedback functionality is currently unsupported.

        Warming up the network with 'acquire_reservoir_states' is generally a good approach if you have a subset of the
        data; those states can then be used in this forecasting algorithm and will be a better representation of the
        data to come than potentially the training data and definitely an array of zeros.

        :param T1: The # of timesteps into the future that the model will try to forecast.
        :param u1: The initial input vector.
        :param x0: The reservoir state at the timestep preceding u1.
        :param continuation: Whether to use the end states from reservoir state acquisition for forecasting.

        :return: The matrix of predictions Y_out w/ shape (K=L, T1); and the final reservoir state X.
        """

        # ----- VALIDATION -----
        if self.W_out is None:
            raise ValueError(f"Network is untrained. W_out has type: None.")
        if self.K != self.L:
            raise ValueError(f"In generative forecasting, the shape of the output must match the input.")
        if T1 <= 0 or not isinstance(T1, int):
            raise ValueError("The forecasting horizon must be a positive integer.")

        # Shape checks if any initial variables were supplied to the method.
        if u1 is not None and u1.shape != (self.K, 1):
            raise ValueError(f"u1 must have shape ({self.K}, 1). Provided shape: {u1.shape}.")
        if x0 is not None and x0.shape != (self.N, 1):
            raise ValueError(f"x0 must have shape ({self.N}, 1). Provided shape: {x0.shape}.")


        # ----- INITIALIZATION -----
        # Allocate memory for the forecasted outputs.
        Y_out = np.zeros(shape=(self.L, T1), dtype=self.dtype)

        # Set the initial input and state.
        if continuation:
            if self.last_state is None or self.last_output is None:
                raise ValueError(
                    "Continuation is enabled, but either the last state or output is missing. Or both...")
            last_state = self.last_state
            input_signal = self.last_output
        else:
            # If continuation is False, there MUST be a starting input point.
            if u1 is None:
                raise ValueError("Method set not to continue from training data, but no starting point was provided.")
            input_signal = u1

            # If continuation is false, it is better to supply an initial_state, but not necessary.
            if x0 is None:
                last_state = np.zeros(shape=(self.N, 1), dtype=self.dtype)
                print("Initial state initialised as a vector of zeros.")
            else:
                last_state = x0


        # ----- FORECASTING -----
        # We divide along biases here since we are constantly getting new inputs.
        if self.bias:
            for t in range(T1):
                # Prepend a bias onto the input_signal.
                input_pattern = np.vstack([np.ones((1, 1)), input_signal])
                # Create new reservoir state.
                current_state = self._update_no_feedback(prev_state=last_state,
                                                         input_pattern=input_pattern)

                # Calculate prediction.
                input_signal = self.W_out @ np.vstack([input_pattern, current_state])

                # Update states:
                last_state = current_state
                Y_out[:, t:t + 1] = input_signal

            return Y_out, last_state


        # No bias included.
        else:
            for t in range(T1):
                # Create new reservoir state.
                current_state = self._update_no_feedback(prev_state=last_state,
                                                         input_pattern=input_signal)

                # Calculate prediction.
                input_signal = self.W_out @ np.vstack([input_signal, current_state])

                # Update states:
                last_state = current_state
                Y_out[:, t:t + 1] = input_signal

            return Y_out, last_state

    @staticmethod
    def print_forecast_metrics(predictions: np.ndarray, targets: np.ndarray):
        """
        Computes and prints forecasting performance metrics if verbosity is enabled.

        :param predictions: Forecasted output with shape (L, T).
        :param targets: Ground truth targets with shape (L, T).

        :return: Pandas DataFrame containing the forecasting metrics.
        """

        rmse = np.sqrt(mean_squared_error(targets, predictions))  # Overall error.
        mae = mean_absolute_error(targets, predictions)  # Depicts absolute deviation
        r2 = r2_score(targets, predictions)  # Goodness of fit.
        max_error = np.max(np.abs(targets - predictions))  # Worst case scenario.
        forecast_bias = np.mean(targets - predictions)  # General over/underestimation.

        # Create DataFrame
        metrics = pd.DataFrame({
            "Metric": ["RMSE", "MAE", "R² Score", "Max Absolute Error", "Forecast Bias"],
            "Value": [rmse, mae, r2, max_error, forecast_bias]
        })


        return metrics

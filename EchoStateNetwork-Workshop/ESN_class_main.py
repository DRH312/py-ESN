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
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M")


"""
This Python file contains the primary classes that constitute a portable echo-state network. 
Some functionality is kept in separate files, including loss functions and random-weight 
generation. The aim is to provide a simple but customisable ESN that can be imported into various
use-cases. 

"""


class EchoStateNetwork:

    def __init__(self, ESN_params: dict, dtype: str = "float64", verbose=0):

        """
        Declaring this class fully initializes an Echo State Network provided that a correct dictionary of inputs is
        supplied to the class.

        :param ESN_params:
        :param dtype:
        :param verbose:
        """

        # Model dimensions:
        self.K = ESN_params['input_dim']  # The number of features in the input vector.
        self.N = ESN_params['nodes']  # The number of neurons in the reservoir.
        self.L = ESN_params['output_dim']  # The number of features in the output vector.

        # Model hyperparameters:
        self.penalty = ESN_params['ridge']
        self.leak = ESN_params['leak']
        self.connectivity = ESN_params['connectivity']
        self.input_scaling = ESN_params['input_scaling']
        self.teacher_scaling = ESN_params['teacher_scaling']
        self.sr = ESN_params['spectral_radius']

        # Parameters that affect the structure of the network.
        self.enable_feedback = ESN_params['enable_forcing']

        # Determines random number generation.
        self.seed = int(ESN_params['seed'])
        self.rng = np.random.default_rng(self.seed)

        # Whether bias is incorporated into the input and output weights.
        self.bias = ESN_params['bias']

        # Cementing the datatype which will be maintained across the whole series of computations.
        self.dtype = dtype
        if self.dtype not in ["float64", "float32", "float16"]:
            raise ValueError(f"Selected datatype must adhere to {['float64', 'float32', 'float16']}")

        # Ensuring that the verbosity scale is an integer, and that it falls into the acceptable range.
        self.verbose = int(verbose)
        if self.verbose > 3:
            self.verbose = 3

        # Model weights:
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.W_fb = None

        # Output a table of the weight matrices dimensions for user

        # Might not keep. Operations should ideally be kept separate from the network's formation.
        # Ideally, the properties of a generally well-trained network should be independent of the data it operates on.
        self.train_length = ESN_params['train_length']
        self.prediction_length = ESN_params['prediction_length']


    def initialize_reservoir(self, distribution: str = 'normal') -> None:

        """
        Initializes an N*N sparse adjacency matrix that defines the reservoir of the ESN. The nonzero elements of the
        matrix can be sampled from either a uniform or standard normal distribution.

        :param distribution:  Distribution to sample the nonzero reservoir weights from.
        :return: csr_matrix: Sparse reservoir matrix with the desired spectral radius.
        """

        # Check that the reservoir distribution requested is a valid input.
        if distribution not in ["uniform", "normal"]:
            raise ValueError(f"Selected distribution must be 'uniform' or 'normal'")

        # Generate sparsity mask. This should be reproducible amongst consistent seeds.
        mask = self.rng.random((self.N, self.N)) < self.connectivity
        num_nonzeros = mask.sum()


        if distribution == 'normal':
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
            self.W_in /= np.abs(self.W_in).max()

            if self.bias:
                self.W_in[:, 0] = 1

            # If feedback is enabled, generated a feedback matrix.
            if self.enable_feedback:
                self.W_fb = self.rng.normal(loc=0, scale=1, size=(self.N, self.L + self.bias)).astype(self.dtype)
                self.W_fb /= np.abs(self.W_fb).max()

                if self.bias:
                    self.W_fb[:, 0] = 1

            else:
                self.W_fb = None


        elif distribution == 'uniform':
            # Sample reservoir weights from a symmetric uniform distribution.

            # Initializing the reservoir adjacency matrix.
            self.W_res = self.rng.uniform(low=-0.5, high=0.5, size=num_nonzeros).astype(self.dtype)
            # Rescale to ensure range is consistent
            self.W_res /= np.abs(self.W_res).max()  # Scale to [-0.5, 0.5]

            # Initializing the input connection weights.
            # These are sampled from the same distribution as the reservoir, but remain dense.
            self.W_in = self.rng.uniform(low=-0.5, high=0.5, size=(self.N, self.K + self.bias)).astype(self.dtype)
            self.W_in /= np.abs(self.W_in).max()  # Scale to [-0.5, 0.5]

            # If feedback is enabled, generated a feedback matrix.
            if self.enable_feedback:
                self.W_fb = self.rng.uniform(low=-0.5, high=0.5, size=(self.N, self.L + self.bias)).astype(self.dtype)
                self.W_fb /= np.abs(self.W_fb).max()

                if self.bias:
                    self.W_fb[:, 0] = 1

            else:
                self.W_fb = None


        row_indices, col_indices = np.where(mask)
        self.W_res = sparse.csr_matrix((self.W_res, (row_indices, col_indices)),
                                       shape=(self.N, self.N), dtype=self.dtype)

        if self.verbose > 0:
            print("Reservoir adjacency matrix initialized. Beginning spectral radius scaling.")

        # Now that matrices are generated, scale the spectral radius of the reservoir.
        self._scale_spectral_radius()

        if self.verbose > 0:
            print("Reservoir weights spectral radius scaling completed.")

        if self.verbose > 1:
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

        if self.verbose > 2:
            print(f"Reservoir Adjacency Matrix is of type {type(self.W_res)} with shape {self.W_res.shape}")

            current_dir = os.getcwd()
            parent_dir = os.path.dirname(current_dir)

            # Define the output directory at the parent level
            output_dir = os.path.join(parent_dir, "Generated_Weights")
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
        print(f"The largest eigenvalue of this matrix was: {largest_eigenvalue}")

        # Scale the sparse reservoir matrix.
        scale_factor = self.sr / largest_eigenvalue
        self.W_res *= scale_factor

        # Forcefully broadening the output distribution.
        # Normalize the non-zero values to [-1, +1].
        # self.W_res.data /= np.abs(self.W_res.data).max()
        # self.W_res.data *= self.sr
        # self.W_res.data *= 0.5

        if self.verbose > 0:
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
            axes[0].hist(self.W_res.data, bins=50, density=True, color='orange', edgecolor='black', alpha=0.75)
            axes[0].set_title("Input Weights Distribution")
            axes[0].set_xlabel("Value")
            axes[0].set_ylabel("Density")
            axes[0].grid(True)

            # Plot histogram for the reservoir weights
            axes[1].hist(self.W_in.flatten(), bins=50, density=True, color='blue', edgecolor='black', alpha=0.75)
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


    def _scale_inputs(self, inputs) -> np.ndarray:
        """
        Applies element-wise scaling to the entire sequence of inputs.

        :param inputs: An input sequence in the form of an array with shape (K, timesteps).
        :return: A scaled input sequence of the same form and shape.
        """

        if inputs.shape[0] != self.K:
            raise ValueError(
                f"Input features ({inputs.shape[0]}) does not match the number of input nodes, ({self.K}).")

        return inputs * self.input_scaling[:, None]


    def _scale_feedback(self, targets) -> np.ndarray:
        """
        Applies feedback scaling factors

        :param targets: A target sequence in the form of an array with shape (L, timesteps).

        :return: A scaled target sequence of the same form and shape.
        """

        if targets.shape[0] != self.L:
            raise ValueError(
                f"Input features ({targets.shape[0]}) does not match the number of input nodes, ({self.L}).")

        return targets * self.teacher_scaling[:, None]


    def _update_no_feedback(self, prev_state, input_pattern) -> np.ndarray:
        """
        Performs a singular reservoir update. No feedback is utilised here.

        :param prev_state: The preceding reservoir state, with shape (N, 1).
        :param input_pattern: The current input vector, with shape (K, 1).

        :return: The current reservoir state, with shape (N, 1) as well.
        """

        nonlinear_contribution = self.W_in @ input_pattern + self.W_res @ prev_state
        return (1 - self.leak) * prev_state + self.leak * np.tanh(nonlinear_contribution)


    def _update_with_feedback(self, prev_state, input_pattern, target) -> np.ndarray:

        """
        Performs a singular reservoir update, including contributions from feedback.

        :param prev_state: The preceding reservoir state, with shape (N, 1)
        :param input_pattern: The current input vector, with shape (K, 1).
        :param target: The current target vector, with shape (L, 1).

        :return: The current reservoir state, with shape (N, 1) as well.
        """

        nonlinear_contribution = self.W_in @ input_pattern + self.W_res @ prev_state + self.W_fb @ target
        return (1 - self.leak) * prev_state + self.leak * np.tanh(nonlinear_contribution)


    def acquire_reservoir_states(self, inputs, teachers=None):

        """
        Perform dimensionality expansion on the data used to train the network. Conventionally, this will usually be
        the past of a signal, for which forecasting is used to predict the signal's evolution. #

        :param inputs: Input sequence with shape (K, timesteps).
        :param teachers: Target sequence with shape (L, timesteps). Only required if feedback is enabled.

        :return: Reservoir states with shape (N, timesteps).
        """

        # Pre-scale the inputs.
        scaled_inputs = self._scale_inputs(inputs) if self.input_scaling is not None else inputs

        # Pre-scale the targets for feedback, but only if feedback is enabled.
        scaled_teachers = None
        if self.enable_feedback:
            if teachers is None:
                raise ValueError("Feedback is enabled but no output sequence has been provided for state acquisition.")
            scaled_teachers = self.teacher_scaling(teachers) if self.teacher_scaling is not None else teachers
            # Prepend a column of zeros to allow for feedback at t=0
            scaled_teachers = np.hstack([np.zeros((self.L, 1), dtype=self.dtype), scaled_teachers])

        # Initialize the base reservoir state, and determine the "length" of the signal.
        timesteps = scaled_inputs.shape[1]
        states = np.zeros(shape=(self.N, timesteps), dtype=self.dtype)

        # Iterate over the timesteps and perform dimensionality expansion to generate reservoir states.
        if self.enable_feedback:
            for t in range(timesteps):
                input_pattern = scaled_inputs[:, t:t+1]  # Maintains the 2D shape of the input.
                teacher_pattern = scaled_teachers[:, t:t + 1]  # Always use y[n-1] for feedback
                states[:, t:t + 1] = self._update_with_feedback(states[:, t - 1:t], input_pattern, teacher_pattern)

        else:
            for t in range(timesteps):
                input_pattern = scaled_inputs[:, t:t+1]  # Maintains the 2D shape of the input.
                states[:, t:t+1] = self._update_no_feedback(states[:, t-1:t], input_pattern)

        return states


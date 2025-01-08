# Essential Libraries
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Accoutrement - some tools lend themselves as being faster for specific operations
from scipy import linalg, sparse


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
        self.iss = ESN_params['input_scaling']
        self.sr = ESN_params['spectral_radius']

        # Parameters that affect the structure of the network.
        self.enable_feedback = ESN_params['enable_forcing']

        # Model weights:
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.W_fb = None

        # Additional parameters:
        self.seed = int(ESN_params['seed'])
        self.rng = np.random.default_rng(self.seed)

        self.dtype = dtype
        if self.dtype not in ["float64", "float32", "float16"]:
            raise ValueError(f"Selected datatype must adhere to {['float64', 'float32', 'float16']}")

        # Ensuring that the verbosity scale is an integer, and that it falls into the acceptable range.
        self.verbose = int(verbose)
        if self.verbose > 3:
            self.verbose = 3

        # Might not keep. Operations should ideally be kept separate from the network's formation.
        # Ideally, the properties of a generally well-trained network should be independent of the data it operates on.
        self.train_length = ESN_params['train_length']
        self.prediction_length = ESN_params['prediction_length']


    def initialize_reservoir(self, distribution: str = 'normal') -> sparse.csr_matrix:

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
        print(mask)


        if distribution == 'normal':
            # Sample reservoir weights from a standard Gaussian distribution.

            # Initializing the reservoir adjacency matrix.
            self.W_res = self.rng.normal(loc=0, scale=1, size=num_nonzeros)
            # Rescale to ensure standard deviation is 1
            self.W_res /= np.std(self.W_res)

            # Initialising the input connection weights.
            # These are sampled from the same distribution as the reservoir, but remain dense.
            self.W_in = self.rng.normal(loc=0, scale=1, size=num_nonzeros)
            # Setting the range to [-1, +1]
            self.W_in /= np.abs(self.W_in).max()


        elif distribution == 'uniform':
            # Sample reservoir weights from a symmetric uniform distribution.

            # Initializing the reservoir adjacency matrix.
            self.W_res = self.rng.uniform(low=-0.5, high=0.5, size=num_nonzeros)
            # Rescale to ensure range is consistent
            self.W_res /= np.abs(self.W_res).max()  # Scale to [-0.5, 0.5]

            # Initializing the input connection weights.
            # These are sampled from the same distribution as the reservoir, but remain dense.
            self.W_in = self.rng.uniform(low=-0.5, high=0.5, size=num_nonzeros)
            self.W_in /= np.abs(self.W_in).max()  # Scale to [-0.5, 0.5]


        row_indices, col_indices = np.where(mask)
        self.W_res = sparse.csr_matrix((self.W_res, (row_indices, col_indices)), shape=(self.N, self.N))

        if self.verbose > 0:
            print("Reservoir adjacency matrix initialized. Beginning spectral radius scaling.")

        # Now that matrices are generated, scale the spectral radius of the reservoir.
        self.scale_spectral_radius()

        if self.verbose > 0:
            print("Reservoir weights initialized.")

        if self.verbose > 1:
            self.plot_reservoir_histogram()

        if self.verbose > 2:
            print(f"Reservoir Adjacency Matrix is of type {type(self.W_res)} with shape {self.W_res.shape}")

        return self. W_res


    def scale_spectral_radius(self) -> None:

        """
        Scales the spectral radius of the reservoir matrix to match user-specification.

        For small reservoirs, (N <= 1000), use 'scipy.sparse.linalg.eigs'.
        For large reservoirs, (N > 1000), use power iteration for efficiency.
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


    def plot_reservoir_histogram(self) -> None:

        """
        Plots a histogram of the reservoir matrix elements to assure the user of correct implementation.
        """

        # Collect only the nonzero values of the reservoir matrix.
        reservoir_values = self.W_res.data
        input_values = self.W_in

        # Checking that there is actually data to plot.
        if len(reservoir_values) == 0:
            raise ValueError("There are no non-zero values in the reservoir matrix to plot.")

        elif len(input_values) == 0:
            raise ValueError("There are no non-zero values in the input connection matrix to plot.")

        else:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Plot histogram for the input weights
            axes[0].hist(input_values, bins=50, density=True, color='orange', edgecolor='black', alpha=0.75)
            axes[0].set_title("Input Weights Distribution")
            axes[0].set_xlabel("Value")
            axes[0].set_ylabel("Density")
            axes[0].grid(True)

            # Plot histogram for the reservoir weights
            axes[1].hist(reservoir_values, bins=50, density=True, color='blue', edgecolor='black', alpha=0.75)
            axes[1].set_title("Reservoir Weights Distribution")
            axes[1].set_xlabel("Value")
            axes[1].set_ylabel("Density")
            axes[1].grid(True)

            # Adjust layout and display the plot
            plt.tight_layout()
            plt.show()

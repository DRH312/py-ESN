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

        # Model weights:
        self.W_in = None
        self.W_res = None
        self.W_out = None
        self.W_fb = None

        # Additional parameters:
        self.dtype = dtype
        if self.dtype not in ["float64", "float32", "float16"]:
            raise ValueError(f"Selected datatype must adhere to {["float64", "float32", "float16"]}")

        self.verbose = int(verbose)
        if self.verbose > 3:
            self.verbose = 3

        self.seed = int(ESN_params['seed'])


        # Might not keep. Operations should ideally be kept separate from the network's formation.
        self.train_length = ESN_params['train_length']
        self.prediction_length = ESN_params['prediction_length']


    def initialize_reservoir(self, distribution: str = 'normal') -> np.ndarray:

        """
        Initializes an N*N sparse adjacency matrix that defines the reservoir of the ESN. The nonzero elements of the
        matrix can be sampled from either a uniform or standard normal distribution.

        :param distribution:  Distribution to sample the nonzero reservoir weights from.
        :return: csr_matrix: Sparse reservoir matrix with the desired spectral radius.
        """

        if distribution not in ["uniform", "normal"]:
            raise ValueError(f"Selected distribution must be 'uniform' or 'normal'")

        # Initializing the random number generator using Numpy.
        rng = np.random.default_rng(self.seed)

        if distribution == 'normal':
            # Sample reservoir weights from a Gaussian distribution.
            self.W_res = sparse.random(
                m=self.N,
                n=self.N,
                density=self.connectivity,
                format='csr',
                data_rvs=lambda n: rng.normal(loc=0, scale=1, size=n)
            )
            # Rescale to ensure standard deviation is 1
            self.W_res.data /= np.std(self.W_res.data)

            if self.verbose > 0:
                print("Base matrix initialized. Beginning spectral radius scaling.")

        elif distribution == 'uniform':
            # Sample reservoir weights from a symmetric normal distribution.
            self.W_res = sparse.random(
                m=self.N,
                n=self.N,
                density=self.connectivity,
                format='csr',
                data_rvs=lambda n: rng.uniform(low=-0.5, high=0.5, size=n)
            )
            # Rescale to ensure range is consistent
            self.W_res.data /= np.abs(self.W_res.data).max()  # Scale to [-0.5, 0.5]

            if self.verbose > 0:
                print("Base matrix initialized. Beginning spectral radius scaling.")

        self.scale_spectral_radius()

        if self.verbose > 0:
            print("Reservoir weights initialized.")

        if self.verbose > 1:
            self.plot_reservoir_histogram()

        if self.verbose > 2:
            print(f"Reservoir Adjacency Matrix is of type {type(self.W_res)} with shape {self.W_res.shape}")



        return self.W_res

    def scale_spectral_radius(self) -> None:

        """
        Scales the spectral radius of the reservoir matrix to match user-specification.

        For small reservoirs, (N <= 1000), use 'scipy.sparse.linalg.eigs'.
        For large reservoirs, (N > 1000), use power iteration for efficiency.
        """

        if self.N < 1001:
            # Use eigs for smaller matrices
            largest_eigenvalue = np.abs(sparse.linalg.eigs(self.W_res, k=1, which='LM', return_eigenvectors=False)[0])

        else:
            largest_eigenvalue = self.estimate_spectral_radius(self.W_res)

        # Scale the sparse reservoir matrix.
        self.W_res *= self.sr / largest_eigenvalue

        if self.verbose > 0:
            print(f"Reservoir spectral radius scaled to: {self.sr}")


    def estimate_spectral_radius(self, max_iter=100, tol=1e-6):

        """
        Uses the power iteration method to estimate the spectral radius of the reservoir matrix, W_res. This method
        is only applied when the reservoir is very large.

        :param max_iter: The maximum number of iterations this method attempts to converge.
        :param tol: The difference between subsequent iterations that defines acceptable convergence.
        :return: The estimated spectral radius of the randomly generated reservoir matrix.
        """

        rng = np.random.default_rng(self.seed)
        v = rng.random(self.N)  # Random vector of size N used to approximate the reservoir's largest eigenvector.

        # Normalize the initial vector. We use Numpy because this vector is dense.
        v /= np.linalg.norm(v)

        # Establish the current estimate for the maximum eigenvalue.
        eig_prev = 0

        for _ in range(max_iter):
            # Multiply v by W_res. Then normalise.
            v = self.W_res @ v
            v /= np.linalg.norm(v)

            # Calculate the corresponding eigenvalue:
            eig = (v.T @ self.W_res @ v) / (v.T @ v)

            # Check for convergence
            if np.abs(eig - eig_prev) < tol:
                return eig

            # Store this current eigenvalue for subsequent steps.
            eig_prev = eig







    def plot_reservoir_histogram(self) -> None:

        """
        Plots a histogram of the reservoir matrix elements to assure the user of correct implementation.
        """

        # Collect only the nonzero values of the reservoir matrix.
        non_zero_values = self.W_res.data

        # Plot histogram.
        plt.hist(non_zero_values, bins=50, density=True, color='blue', edgecolor='black', alpha=0.75)
        plt.title("Distribution of reservoir's nonzero elements")
        plt.xlabel("Value")
        plt.ylabel("Density")
        plt.grid(True)
        plt.show()

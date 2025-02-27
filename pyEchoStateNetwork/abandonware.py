# Imports
import numpy as np


"""
This file contains WIP functionalities for the ESN class within the same document. These methods were ineffective for 
their purpose and are hence shelved until they can be fixed.
"""


def estimate_spectral_radius(self, max_iter=1000, tol=1e-6):
    """
    Uses the power iteration method to estimate the spectral radius of the reservoir matrix, W_res. This method
    is only applied when the reservoir is very large. When re-introdued to the class, it should be done so
    conditionally during spectral radius scaling depending on the size of the reservoir.

    :param max_iter: The maximum number of iterations this method attempts to converge.
    :param tol: The difference between subsequent iterations that defines acceptable convergence.
    :return: The estimated spectral radius of the randomly generated reservoir matrix.
    """

    # Random vector of size N used to approximate the reservoir's largest eigenvector.
    v = self.rng.normal(size=self.N)

    # Normalize the initial vector. We use Numpy because this vector is dense.
    v /= np.linalg.norm(v)

    # Establish the current estimate for the maximum eigenvalue.
    eig_prev = 0

    for _ in range(max_iter):
        # Multiply v by W_res. Then normalise.
        v = self.W_res @ v
        v /= np.linalg.norm(v)

        # Calculate the corresponding eigenvalue:
        # (The vector is already normalised, so we do not need to divide by (v.t @ v) )
        eig = (v.T @ self.W_res @ v) / (v.T @ v)

        # Check for convergence
        if np.abs(eig - eig_prev) < tol:
            if self.verbosity > 0:
                print(f"Converged to spectral radius: {eig:.6f} in {_ + 1} iterations.")
            return eig

        # Store this current eigenvalue for subsequent steps.
        eig_prev = eig

    if self.verbosity > 0:
        print(f"Power iteration failed to converge within {max_iter} iterations. Returning last estimate {eig:.6f}")
    return eig


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


    def _scale_teachers(self, targets) -> np.ndarray:
        """
        Applies feedback scaling factors

        :param targets: A target sequence in the form of an array with shape (L, timesteps).

        :return: A scaled target sequence of the same form and shape.
        """

        if targets.shape[0] != self.L:
            raise ValueError(
                f"Input features ({targets.shape[0]}) does not match the number of input nodes, ({self.L}).")

        return targets * self.teacher_scaling[:, None]
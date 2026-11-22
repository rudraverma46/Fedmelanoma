# earth_mover_distance.py

import numpy as np
from scipy.optimize import linprog


class EarthMoverDistance:

    # Semantic Cost Matrix for Imaging Modality
    MODALITY_COST_MATRIX = np.array([
        [0.00, 0.30, 0.70],
        [0.30, 0.00, 0.45],
        [0.70, 0.45, 0.00]
    ])

    @staticmethod
    def wasserstein_1d(global_dist, client_dist):
        """
        Fitzpatrick diversity using 1D Wasserstein distance.

        W1(P,Q) = sum |CDF(P) - CDF(Q)|
        """

        global_cdf = np.cumsum(global_dist)
        client_cdf = np.cumsum(client_dist)

        return np.sum(np.abs(global_cdf - client_cdf))

    @staticmethod
    def optimal_transport_distance(source, target, cost_matrix):
        """
        Compute Earth Mover Distance (Optimal Transport)
        using Linear Programming.
        """

        source = np.asarray(source, dtype=float)
        target = np.asarray(target, dtype=float)

        n = len(source)
        m = len(target)

        c = cost_matrix.flatten()

        A_eq = []
        b_eq = []

        # Source constraints
        for i in range(n):
            row = np.zeros(n * m)
            for j in range(m):
                row[i * m + j] = 1
            A_eq.append(row)
            b_eq.append(source[i])

        # Target constraints
        for j in range(m):
            row = np.zeros(n * m)
            for i in range(n):
                row[i * m + j] = 1
            A_eq.append(row)
            b_eq.append(target[j])

        bounds = [(0, None)] * (n * m)

        result = linprog(
            c,
            A_eq=np.array(A_eq),
            b_eq=np.array(b_eq),
            bounds=bounds,
            method="highs"
        )

        if not result.success:
            raise ValueError(
                f"Optimal transport failed: {result.message}"
            )

        return result.fun

    @classmethod
    def compute_diversity_score(
        cls,
        global_fitzpatrick,
        client_fitzpatrick,
        global_modality,
        client_modality,
        fitz_weight=0.40,
        modality_weight=0.35
    ):
        """
        Computes raw diversity score for a client from skin type and imaging modality.

        The confirmation-type axis (weight 0.25) is removed because diagnosis_confirm_type
        leaks the label; D_i is max-normalised afterwards, so the remaining weights act as
        0.40 / 0.75 and 0.35 / 0.75.
        """

        fitz_distance = cls.wasserstein_1d(
            global_fitzpatrick,
            client_fitzpatrick
        )

        modality_distance = cls.optimal_transport_distance(
            global_modality,
            client_modality,
            cls.MODALITY_COST_MATRIX
        )

        diversity_raw = (
            fitz_weight * fitz_distance
            + modality_weight * modality_distance
        )

        return {
            "fitzpatrick_distance": fitz_distance,
            "modality_distance": modality_distance,
            "raw_diversity_score": diversity_raw
        }
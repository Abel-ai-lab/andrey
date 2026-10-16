"""Per-algorithm comparison spec: structural vs numeric fields, and tolerances.

Each algorithm declares ``structural`` fields (compared exactly) and ``numeric``
fields (compared within ``atol``/``rtol``). Tolerances are provisional defaults.
"""

from __future__ import annotations

SPEC: dict[str, dict] = {
    "DirectLiNGAM": {
        "structural": ["causal_order"],
        "numeric": {"weighted_adjacency": {"atol": 1e-6, "rtol": 1e-5}},
    },
    "PC": {
        "structural": ["graph"],
        "numeric": {},
    },
    "GES": {
        "structural": ["graph"],
        "numeric": {"score": {"atol": 1e-6, "rtol": 1e-6}},
    },
    "GIES": {
        "structural": ["graph"],
        "numeric": {"score": {"atol": 1e-6, "rtol": 1e-6}},
    },
    "XGES": {
        "structural": ["graph"],
        "numeric": {"score": {"atol": 1e-6, "rtol": 1e-6}},
    },
    "HC": {
        "structural": ["graph"],
        "numeric": {"score": {"atol": 1e-6, "rtol": 1e-6}},
    },
    "Tabu": {
        "structural": ["graph"],
        "numeric": {"score": {"atol": 1e-6, "rtol": 1e-6}},
    },
    "ExactSearch": {
        "structural": ["graph"],
        "numeric": {},
    },
    "FCI": {
        "structural": ["graph"],
        "numeric": {},
    },
    "GFCI": {
        "structural": ["graph"],
        "numeric": {},
    },
    "CDNOD": {
        "structural": ["graph"],
        "numeric": {},
    },
    "BOSS": {
        "structural": ["graph"],
        "numeric": {},
    },
    "GRaSP": {
        "structural": ["graph"],
        "numeric": {},
    },
    "ICALiNGAM": {
        "structural": ["causal_order"],
        "numeric": {"weighted_adjacency": {"atol": 1e-6, "rtol": 1e-5}},
    },
    # Multi-group: one shared causal order (structural) + a stack of per-group weighted matrices
    # (numeric, shape (n_groups, d, d) -- compared element-wise within tolerance).
    "MultiGroupDirectLiNGAM": {
        "structural": ["causal_order"],
        "numeric": {"group_weighted_adjacency": {"atol": 1e-6, "rtol": 1e-5}},
    },
    # --- Temporal (lagged) methods: causal order is the cross-source slice gate.
    "VARLiNGAM": {
        "structural": ["causal_order"],
        "numeric": {"adjacency_matrices": {"atol": 1e-6, "rtol": 1e-5}},
    },
    "VARMALiNGAM": {
        "structural": ["causal_order"],
        "numeric": {
            "psis": {"atol": 1e-6, "rtol": 1e-5},
            "omegas": {"atol": 1e-6, "rtol": 1e-5},
        },
    },
    # PCMCI, PCMCI+, LPCMCI: the per-lag graphs (LPCMCI's links with both marks) and the
    # conflicts, exactly; the test p-values and statistics within rounding.
    "PCMCI": {
        "structural": ["graph"],
        "numeric": {
            "p_values": {"atol": 1e-8, "rtol": 1e-6},
            "statistics": {"atol": 1e-8, "rtol": 1e-6},
        },
    },
    "PCMCIplus": {
        "structural": ["graph", "conflicts"],
        "numeric": {
            "p_values": {"atol": 1e-8, "rtol": 1e-6},
            "statistics": {"atol": 1e-8, "rtol": 1e-6},
        },
    },
    "LPCMCI": {
        "structural": ["links", "conflicts"],
        "numeric": {
            "p_values": {"atol": 1e-8, "rtol": 1e-6},
            "statistics": {"atol": 1e-8, "rtol": 1e-6},
        },
    },
    # Longitudinal LiNGAM: each time point's causal order and every coefficient block, NaN where a
    # block reaches before the panel start.
    "LongitudinalLiNGAM": {
        "structural": ["causal_orders"],
        "numeric": {"time_weights": {"atol": 1e-6, "rtol": 1e-5}},
    },
    "VARMALiNGAMPruned": {
        "structural": ["causal_order"],
        "numeric": {
            "psis": {"atol": 1e-6, "rtol": 1e-5},
            "omegas": {"atol": 1e-6, "rtol": 1e-5},
        },
    },
    # Granger has no causal order; its structural gate is the binary support.
    "Granger": {
        "structural": ["adj"],
        "numeric": {"coeff": {"atol": 1e-6, "rtol": 1e-5}},
    },
    # --- Irregular / pairwise & latent-confounder methods.
    # ANM/PNL: the decided two-node graph, and both p-values and HSIC statistics.
    # The Gaussian-process hyperparameter search ends within rounding of its optimum, so ANM's
    # p-values and statistics get a looser tolerance than PNL's.
    "ANM": {
        "structural": ["graph"],
        "numeric": {
            "pval_forward": {"atol": 1e-4, "rtol": 1e-4},
            "pval_backward": {"atol": 1e-4, "rtol": 1e-4},
            "stat_forward": {"atol": 1e-4, "rtol": 1e-4},
            "stat_backward": {"atol": 1e-4, "rtol": 1e-4},
        },
    },
    "PNL": {
        "structural": ["graph"],
        "numeric": {
            "pval_forward": {"atol": 1e-6, "rtol": 1e-5},
            "pval_backward": {"atol": 1e-6, "rtol": 1e-5},
            "stat_forward": {"atol": 1e-6, "rtol": 1e-5},
            "stat_backward": {"atol": 1e-6, "rtol": 1e-5},
        },
    },
    "RCD": {
        "structural": ["graph"],
        "numeric": {"weighted_adjacency": {"atol": 1e-6, "rtol": 1e-5}},
    },
    "BottomUpParceLiNGAM": {
        "structural": ["graph", "causal_order"],
        "numeric": {"weighted_adjacency": {"atol": 1e-6, "rtol": 1e-5}},
    },
    # --- Nonlinear conditional-independence tests: the p-values of the case's queries.
    "KCI": {"structural": [], "numeric": {"pvalues": {"atol": 1e-6, "rtol": 1e-5}}},
    "RCoT": {"structural": [], "numeric": {"pvalues": {"atol": 1e-6, "rtol": 1e-5}}},
    "CMIknn": {"structural": [], "numeric": {"pvalues": {"atol": 1e-6, "rtol": 1e-5}}},
    "GPDC": {"structural": [], "numeric": {"pvalues": {"atol": 1e-6, "rtol": 1e-5}}},
}

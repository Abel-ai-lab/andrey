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
    "HC": {
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
    # Granger has no causal order; its structural gate is the binary support.
    "Granger": {
        "structural": ["adj"],
        "numeric": {"coeff": {"atol": 1e-6, "rtol": 1e-5}},
    },
    # --- Irregular / pairwise & latent-confounder methods.
    # ANM/PNL payload is the two p-values (numeric); no structural graph.
}

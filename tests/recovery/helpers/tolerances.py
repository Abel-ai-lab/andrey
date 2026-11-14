"""Per-algorithm comparison spec: structural vs numeric fields, and tolerances.

Each algorithm declares ``structural`` fields (compared exactly) and ``numeric``
fields (compared within ``atol``/``rtol``). Tolerances are provisional defaults.
"""

from __future__ import annotations

#: The graph alone, exactly.
GRAPH_ONLY = {"structural": ["graph"], "numeric": {}}
#: The graph exactly and the objective score within rounding.
GRAPH_SCORE = {"structural": ["graph"], "numeric": {"score": {"atol": 1e-6, "rtol": 1e-6}}}

SPEC: dict[str, dict] = {
    "DirectLiNGAM": {
        "structural": ["causal_order"],
        "numeric": {"weighted_adjacency": {"atol": 1e-6, "rtol": 1e-5}},
    },
    "PC": GRAPH_ONLY,
    "GES": GRAPH_SCORE,
    "GIES": GRAPH_SCORE,
    "GnIES": {
        "structural": ["graph", "targets"],
        "numeric": {"score": {"atol": 1e-6, "rtol": 1e-6}},
    },
    # IGSP: the graph and its objective, the edges and the I-contradicting edges, all counts.
    "IGSP": {
        "structural": ["graph", "score", "n_contradicting"],
        "numeric": {},
    },
    "XGES": GRAPH_SCORE,
    "AGES": GRAPH_ONLY,
    "HC": GRAPH_SCORE,
    "Tabu": GRAPH_SCORE,
    "IAMB": GRAPH_ONLY,
    "InterIAMB": GRAPH_ONLY,
    "GS": GRAPH_ONLY,
    "FastIAMB": GRAPH_ONLY,
    "MMPC": GRAPH_ONLY,
    "HITONPC": GRAPH_ONLY,
    "PCMB": GRAPH_ONLY,
    "HITONMB": GRAPH_ONLY,
    "MMMB": GRAPH_ONLY,
    # Y-structures: the claimed pairs in their listed order, and the scores within rounding.
    "YStructures": {
        "structural": ["graph", "claims"],
        "numeric": {"scores": {"atol": 1e-6, "rtol": 1e-6}},
    },
    "CMB": GRAPH_ONLY,
    "MMBbyMMB": GRAPH_ONLY,
    "MMHC": GRAPH_SCORE,
    "RSMAX2": GRAPH_SCORE,
    "ExactSearch": GRAPH_ONLY,
    "FCI": GRAPH_ONLY,
    "GFCI": GRAPH_ONLY,
    "BOSSFCI": GRAPH_ONLY,
    "GRaSPFCI": GRAPH_ONLY,
    # CAM records no weights: its DAG and its causal order.
    "CAM": {
        "structural": ["graph", "causal_order"],
        "numeric": {},
    },
    # CALM's weights come from a torch optimizer, so only the graph is recorded.
    "CALM": GRAPH_ONLY,
    "CDNOD": GRAPH_ONLY,
    # FCI-JCI: the PAG over the data and the index's edges.
    "FCIJCI": {"structural": ["graph", "index_edges"], "numeric": {}},
    # ICP: the DAG into the target and the accepted sets exactly; the p-values, NaN at the target,
    # within rounding.
    "ICP": {
        "structural": ["graph", "accepted_sets"],
        "numeric": {"p_values": {"atol": 1e-8, "rtol": 1e-6}},
    },
    "BOSS": GRAPH_ONLY,
    "GRaSP": GRAPH_ONLY,
    "UT-IGSP": {
        "structural": ["graph", "score", "targets"],
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
    # J-PCMCI+: PCMCI+'s fields over the system variables, and the context and dummy links.
    "JPCMCIplus": {
        "structural": ["graph", "conflicts", "context_links", "index_edges", "time_index_edges"],
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
    # DYNOTEARS: the per-lag graphs exactly; the weights within what L-BFGS-B's stopping rule
    # leaves open: a change of the input in its last bits moved them by up to 1e-3.
    "DYNOTEARS": {
        "structural": ["graph"],
        "numeric": {"lag_weights": {"atol": 5e-3, "rtol": 0.0}},
    },
    # NOTEARS: the DAG exactly; the weights within what L-BFGS-B's stopping rule leaves open: a
    # change of the input in its last bits moved them by up to 2.3e-3 over 100 refits.
    "NOTEARS": {
        "structural": ["graph"],
        "numeric": {"weighted_adjacency": {"atol": 1e-2, "rtol": 0.0}},
    },
    # Granger has no causal order: the per-lag graphs exactly, the weights within rounding.
    "Granger": {
        "structural": ["graph"],
        "numeric": {"lag_weights": {"atol": 1e-6, "rtol": 1e-5}},
    },
    # CAM-UV: the PAG's marks and the pairs it marks through a hidden variable, both exact.
    "CAMUV": {
        "structural": ["graph", "latent_pairs"],
        "numeric": {},
    },
    # RESIT: the DAG and the causal order of its sink search, both exact.
    "RESIT": {
        "structural": ["graph", "causal_order"],
        "numeric": {},
    },
    # SCORE: the DAG and the causal order of its leaf search, both exact.
    "ScoreMatching": {
        "structural": ["graph", "causal_order"],
        "numeric": {},
    },
    # NoGAM: the DAG and the causal order of its leaf search, both exact.
    "NoGAM": {
        "structural": ["graph", "causal_order"],
        "numeric": {},
    },
    # GIN: the latent graph and the clusters in causal order, both exact.
    "GIN": {
        "structural": ["graph", "causal_order"],
        "numeric": {},
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
    # LOCI: the decided graph, both p-values and statistics, and both models' log-likelihoods. The
    # Newton fit stops on a change of 1e-6 in its mean objective in a full step, so the fit's
    # numbers get ANM's tolerance. A p-value can lie far below any absolute tolerance, so the two
    # p-values are compared relatively, which bounds the change of their logarithms.
    "LOCI": {
        "structural": ["graph"],
        "numeric": {
            **{field: {"atol": 0.0, "rtol": 1e-4} for field in ("pval_forward", "pval_backward")},
            **{
                field: {"atol": 1e-4, "rtol": 1e-4}
                for field in ("stat_forward", "stat_backward", "ll_forward", "ll_backward")
            },
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

"""The homepage's content, generated from the release summaries and the method registry.

``docs/conf.py`` fills the homepage template (``site/homepage.html``) with
:func:`homepage_context`. ``docs/DEVELOPMENT.md`` has the build command.

Every number on the homepage comes from a release's published benchmark summary (``RELEASES``) or
from the method registry. A method measured against the same package in two summaries shows the
newer summary's rows (:func:`newest`). Each registry family shows one method, the one
:func:`featured_methods` picks, timed against every package compared with it, beside the outcome
its release card records. The README, the live demo, and the homepage's description state
``CLAIM``, which carries no number; :func:`claim_sentence` words the first summary's headline. The
live demo's charts and tables (``apps/build_assets.py``) and the README's claims
(``apps/build_launch.py``) come from the same functions. A summary outside
``benchmarks/published/`` loads only as a placeholder, which marks the page as not for release.
Links are relative to the site root.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import re
import runpy
import tempfile
import xml.etree.ElementTree as ET
from collections import Counter
from html import escape
from pathlib import Path

import networkx as nx
import numpy as np
import yaml
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

import andrey.viz
from andrey import GraphStructure
from andrey.cli import main as andrey_cli
from andrey.spec import DATA_KINDS, get_spec, list_specs

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
DOCS = ROOT / "docs"
PUBLISHED = ROOT / "benchmarks" / "published"
# Every release's published comparison with other packages, oldest first, by folder under
# PUBLISHED. The speed and accuracy sections cover them all. The first is SUMMARY, the one a
# placeholder preview replaces.
RELEASES = (
    "0.1.0",
    "0.2.0",
    "0.3.0",
    "0.4.0",
    "0.4.1",
    "0.5.0",
    "0.6.0",
    "0.7.0",
    "0.8.0",
    "0.9.0",
    "0.10.0",
    "0.11.0",
    "0.12.0",
    "0.13.0",
    "0.14.0",
    "0.15.0",
    "0.16.0",
    "0.17.0",
)
SUMMARY = PUBLISHED / RELEASES[0] / "summary.json"

palette_block = runpy.run_path(str(DOCS / "palette_css.py"))["palette_block"]
# The number formats the live demo shares.
_formatting = runpy.run_path(str(SITE / "formatting.py"))
# Each art's description, its images' alt text (site/covers.py).
cover_alt = runpy.run_path(str(SITE / "covers.py"))["cover_alt"]
ratio_text, seconds_text, number_text = (
    _formatting[name] for name in ("ratio_text", "seconds_text", "number_text")
)

# The method registry's own families, in the order the method list shows them.
SPEC_FAMILIES = {
    "constraint": "Constraint-based",
    "score": "Score-based",
    "permutation": "Permutation-based",
    "functional": "Functional causal models",
    "temporal": "Temporal causal discovery",
    "latent": "Latent variables",
    "pairwise": "Pairwise direction",
}
REGIMES = {
    "linear_gauss_er": "linear-Gaussian data from Erdos-Renyi graphs",
    "linear_gauss_er_unitless": "linear-Gaussian data from Erdos-Renyi graphs, not standardized",
    "linear_gauss_er_d200": "linear-Gaussian data from Erdos-Renyi graphs with 200 variables",
    "latent_gauss_er": "linear-Gaussian data from Erdos-Renyi graphs with hidden confounders",
    "lingam_sf": "linear non-Gaussian data from scale-free graphs",
    "latent_lingam_er": "linear non-Gaussian data from Erdos-Renyi graphs with hidden confounders",
    "anm_pair": "cause-effect pairs with additive noise",
    "pnl_pair": "cause-effect pairs with post-nonlinear noise",
    "pcmci_paper": "lagged linear-Gaussian time series",
    "pcmciplus_paper": "linear-Gaussian time series with same-step links",
    "lpcmci_paper": "linear-Gaussian time series with unobserved series",
    "varma_lingam_paper": "linear non-Gaussian time series with a moving average",
    "longitudinal_lingam_paper": "linear non-Gaussian longitudinal data",
    "var2_lagged": "lagged linear-Gaussian time series",
    "anm_er": "additive-noise data from Erdos-Renyi graphs",
    "nonmonotone_er": "nonmonotone data from Erdos-Renyi graphs",
    "diabetes": "categorical data from the diabetes network",
    "mixed_er": "mixed continuous and categorical data from Erdos-Renyi graphs",
    "discrete_er": "categorical data from Erdos-Renyi graphs",
    "discrete_networks_chisq": "categorical data from the asia, child, insurance, alarm, and "
    "hepar2 networks, tested with the chi-square test",
    "discrete_networks_gsq": "categorical data from the asia, child, insurance, alarm, and "
    "hepar2 networks, tested with the G-square test",
    "interventional_gauss_er": "linear-Gaussian data from Erdos-Renyi graphs with interventions",
    "multi_group_er": "linear non-Gaussian data in three groups sharing one Erdos-Renyi graph",
    "latent_measurement": "linear data with latent variables, each measured by three indicators",
    "domain_shift_er": "linear-Gaussian data from Erdos-Renyi graphs whose mechanisms shift "
    "across domains",
    "fcm_anm_er": "additive-noise data with Gaussian noise from Erdos-Renyi graphs, standardized, "
    "in the method's claim",
    "fcm_anm_gumbel_er": "additive-noise data with Gumbel noise from Erdos-Renyi graphs, "
    "standardized, in the method's claim",
    "fcm_anm_er_raw": "additive-noise data with Gaussian noise from Erdos-Renyi graphs, not "
    "standardized, in the method's claim",
    "fcm_anm_gumbel_er_raw": "additive-noise data with Gumbel noise from Erdos-Renyi graphs, not "
    "standardized, in the method's claim",
    "fcm_lsnm_er": "location-scale noise data from Erdos-Renyi graphs, standardized, outside the "
    "method's claim",
    "fcm_lsnm_er_raw": "location-scale noise data from Erdos-Renyi graphs, not standardized, "
    "outside the method's claim",
    "fcm_linear_gauss_er": "linear-Gaussian data from Erdos-Renyi graphs, standardized, outside "
    "the method's claim",
    "fcm_linear_gauss_er_raw": "linear-Gaussian data from Erdos-Renyi graphs, not standardized, "
    "outside the method's claim",
    "icp_appendix_g": "linear-Gaussian data in two environments, with noise interventions on "
    "variables other than the target",
    "icp_appendix_g_target": "linear-Gaussian data in two environments, with noise interventions "
    "that can hit the target, outside the method's claim",
    "icp_do": "linear-Gaussian data from observation and do-interventions on two or three "
    "variables other than the target",
    "icp_appendix_g_capped": "linear-Gaussian data in two environments with 16 or 21 variables, "
    "the target's parents capped at three",
    "icp_speed": "linear-Gaussian data in two environments with 9 to 15 variables",
    "gnies_a": "linear-Gaussian data from Erdos-Renyi graphs, observation and three noise "
    "interventions of one target each",
    "gnies_b": "linear-Gaussian data from Erdos-Renyi graphs at mean degree 4, observation and "
    "three noise interventions of one target each",
    "gnies_c": "linear-Gaussian data from Erdos-Renyi graphs, observation and three noise "
    "interventions of three targets each",
    "gnies_d": "linear-Gaussian data from Erdos-Renyi graphs, observation and three weaker noise "
    "interventions of one target each",
    "gnies_e": "linear-Gaussian data from Erdos-Renyi graphs, four noise interventions of one "
    "target each and no observation",
    "gnies_f": "linear-Gaussian data from Erdos-Renyi graphs, observation and nine noise "
    "interventions of one target each",
    "gnies_g": "linear-Gaussian data from Erdos-Renyi graphs, 1,000 observed rows and three noise "
    "interventions of 100 rows, one target each",
    "gnies_h1": "linear-Gaussian data from Erdos-Renyi graphs, observation and one noise "
    "intervention of one target",
    "gnies_h2": "linear-Gaussian data from Erdos-Renyi graphs, observation and two noise "
    "interventions of one target each",
    "gnies_s": "linear-Gaussian data from Erdos-Renyi graphs with signed weights and unit noise, "
    "observation and three noise interventions of one target each",
    "gnies_hard": "linear-Gaussian data from Erdos-Renyi graphs, observation and three hard "
    "interventions of one target each, outside the method's claim",
    "gnies_mean_shift": "linear-Gaussian data from Erdos-Renyi graphs, observation and three mean "
    "shifts of one target each, outside the method's claim",
    "gnies_weights_halved": "linear-Gaussian data from Erdos-Renyi graphs, observation and three "
    "interventions that halve one target's incoming weights, outside the method's claim",
    "gnies_laplace": "linear data with Laplace noise from Erdos-Renyi graphs, observation and "
    "three noise interventions of one target each, outside the method's claim",
    "gnies_nonlinear": "additive nonlinear data from Erdos-Renyi graphs, observation and three "
    "noise interventions of one target each, outside the method's claim",
    "gnies_variables": "linear-Gaussian data from Erdos-Renyi graphs with 20 to 100 variables, "
    "observation and three noise interventions of one target each, outside the method's claim",
    "ut_igsp_g1_shift_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with no off-targets",
    "ut_igsp_g1_shift_l1": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with one off-target each",
    "ut_igsp_g1_shift_l2": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with two off-targets each",
    "ut_igsp_g1_shift_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with three off-targets each",
    "ut_igsp_g2_perfect_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that cut one target from its causes, with no off-targets",
    "ut_igsp_g2_perfect_l1": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that cut one target from its causes, with one off-target each",
    "ut_igsp_g2_perfect_l2": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that cut one target from its causes, with two off-targets each",
    "ut_igsp_g2_perfect_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that cut one target from its causes, with three off-targets each",
    "ut_igsp_g3_perfect_code_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and "
    "five interventions that cut one target from its causes and nearly fix it, with no off-targets",
    "ut_igsp_g3_perfect_code_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and "
    "five interventions that cut one target from its causes and nearly fix it, with three "
    "off-targets each",
    "ut_igsp_g3_shift2_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 2, with no off-targets",
    "ut_igsp_g3_shift2_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 2, with three off-targets each",
    "ut_igsp_g4_variance_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that double one target's noise, with no off-targets",
    "ut_igsp_g4_variance_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that double one target's noise, with three off-targets each",
    "ut_igsp_g5_shift_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with no off-targets, and no targets given",
    "ut_igsp_g5_shift_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with three off-targets each, and no targets "
    "given",
    "ut_igsp_g6_weight_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that halve the weights into one target, with no off-targets",
    "ut_igsp_g6_weight_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that halve the weights into one target, with three off-targets each",
    "ut_igsp_o1_shift_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with no off-targets, few rows per "
    "intervention, outside the method's claim",
    "ut_igsp_o1_shift_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with three off-targets each, few rows per "
    "intervention, outside the method's claim",
    "ut_igsp_o2_weak0p25_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 0.25, with no off-targets, outside the method's "
    "claim",
    "ut_igsp_o2_weak0p25_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 0.25, with three off-targets each, outside the "
    "method's claim",
    "ut_igsp_o2_weak0p5_l0": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 0.5, with no off-targets, outside the method's "
    "claim",
    "ut_igsp_o2_weak0p5_l3": "linear-Gaussian data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 0.5, with three off-targets each, outside the "
    "method's claim",
    "ut_igsp_o3_perfect_l0_deg5_k3": "linear-Gaussian data from Erdos-Renyi graphs with 50 "
    "variables at mean degree 5, observation and three interventions that cut one target from its "
    "causes, outside the method's claim",
    "ut_igsp_o4_shift_l3": "linear-Gaussian data from Erdos-Renyi graphs with 50 and 100 "
    "variables, observation and five interventions that shift one target's mean by 1, with three "
    "off-targets each, outside the method's claim",
    "ut_igsp_o5_shift_l0": "additive nonlinear data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with no off-targets, Gaussian tests, outside "
    "the method's claim",
    "ut_igsp_o5_shift_l3": "additive nonlinear data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with three off-targets each, Gaussian tests, "
    "outside the method's claim",
    "ut_igsp_o6_shift_l0": "additive nonlinear data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with no off-targets, kernel tests, outside "
    "the method's claim",
    "ut_igsp_o6_shift_l3": "additive nonlinear data from Erdos-Renyi graphs, observation and five "
    "interventions that shift one target's mean by 1, with three off-targets each, kernel tests, "
    "outside the method's claim",
    "ut_igsp_o8_shift_l0_k1": "linear-Gaussian data from Erdos-Renyi graphs, observation and one "
    "intervention that shifts one target's mean by 1, with no off-targets, outside the method's "
    "claim",
    "ut_igsp_o8_shift_l0_k2": "linear-Gaussian data from Erdos-Renyi graphs, observation and two "
    "interventions that shift one target's mean by 1, with no off-targets, outside the method's "
    "claim",
    "ut_igsp_o8_shift_l3_k1": "linear-Gaussian data from Erdos-Renyi graphs, observation and one "
    "intervention that shifts one target's mean by 1, with three off-targets each, outside the "
    "method's claim",
    "ut_igsp_o8_shift_l3_k2": "linear-Gaussian data from Erdos-Renyi graphs, observation and two "
    "interventions that shift one target's mean by 1, with three off-targets each, outside the "
    "method's claim",
    "ut_igsp_o9_shift_l0_obs2000": "linear-Gaussian data from Erdos-Renyi graphs, 2,000 observed "
    "rows and five interventions of 300 rows that shift one target's mean by 1, with no "
    "off-targets, outside the method's claim",
    "ut_igsp_o9_shift_l3_obs2000": "linear-Gaussian data from Erdos-Renyi graphs, 2,000 observed "
    "rows and five interventions of 300 rows that shift one target's mean by 1, with three "
    "off-targets each, outside the method's claim",
    "igsp_s_family45_c0p1": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "weights at least 0.1 in size, observation and one intervention on variables 4 and 5",
    "igsp_s_family45_c0p25": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "weights at least 0.25 in size, observation and one intervention on variables 4 and 5",
    "igsp_s_family45_c0p5": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "weights at least 0.5 in size, observation and one intervention on variables 4 and 5",
    "igsp_s_family4_c0p1": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "weights at least 0.1 in size, observation and one intervention on variable 4",
    "igsp_s_family4_c0p25": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "weights at least 0.25 in size, observation and one intervention on variable 4",
    "igsp_s_family4_c0p5": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "weights at least 0.5 in size, observation and one intervention on variable 4",
    "igsp_s_family5_c0p1": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "weights at least 0.1 in size, observation and one intervention on variable 5",
    "igsp_s_family5_c0p25": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "weights at least 0.25 in size, observation and one intervention on variable 5",
    "igsp_s_family5_c0p5": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "weights at least 0.5 in size, observation and one intervention on variable 5",
    "igsp_w_k1": "linear-Gaussian data from Erdos-Renyi graphs, observation and one intervention "
    "on five variables, ten at 20 variables",
    "igsp_w_k2": "linear-Gaussian data from Erdos-Renyi graphs, observation and two interventions "
    "on four variables each, eight at 20 variables",
    "igsp_y5_imperfect": "linear-Gaussian data from Erdos-Renyi graphs with 20 variables, "
    "observation and one intervention per variable, each cutting it from its causes on half the "
    "rows",
    "igsp_y5_inhibiting": "linear-Gaussian data from Erdos-Renyi graphs with 20 variables, "
    "observation and one intervention per variable, each dividing the weights into it by 10",
    "igsp_y5_perfect": "linear-Gaussian data from Erdos-Renyi graphs with 20 variables, "
    "observation and one intervention per variable, each cutting it from its causes",
    "igsp_y6_imperfect_k1": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and one intervention of one target, each cutting it from its causes on half the "
    "rows",
    "igsp_y6_imperfect_k10": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and one intervention per variable, each cutting it from its causes on half the "
    "rows",
    "igsp_y6_imperfect_k2": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and two interventions of one target, each cutting it from its causes on half the "
    "rows",
    "igsp_y6_imperfect_k4": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and four interventions of one target, each cutting it from its causes on half the "
    "rows",
    "igsp_y6_imperfect_k6": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and six interventions of one target, each cutting it from its causes on half the "
    "rows",
    "igsp_y6_imperfect_k8": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and eight interventions of one target, each cutting it from its causes on half "
    "the rows",
    "igsp_y6_inhibiting_k1": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and one intervention of one target, each dividing the weights into it by 10",
    "igsp_y6_inhibiting_k10": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and one intervention per variable, each dividing the weights into it by 10",
    "igsp_y6_inhibiting_k2": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and two interventions of one target, each dividing the weights into it by 10",
    "igsp_y6_inhibiting_k4": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and four interventions of one target, each dividing the weights into it by 10",
    "igsp_y6_inhibiting_k6": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and six interventions of one target, each dividing the weights into it by 10",
    "igsp_y6_inhibiting_k8": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and eight interventions of one target, each dividing the weights into it by 10",
    "igsp_y6_perfect_k1": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and one intervention of one target, each cutting it from its causes",
    "igsp_y6_perfect_k10": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and one intervention per variable, each cutting it from its causes",
    "igsp_y6_perfect_k2": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and two interventions of one target, each cutting it from its causes",
    "igsp_y6_perfect_k4": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and four interventions of one target, each cutting it from its causes",
    "igsp_y6_perfect_k6": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and six interventions of one target, each cutting it from its causes",
    "igsp_y6_perfect_k8": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and eight interventions of one target, each cutting it from its causes",
    "igsp_y8_imperfect": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and one intervention per pair of variables, each cutting both from their causes "
    "on half the rows",
    "igsp_y8_inhibiting": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and one intervention per pair of variables, each dividing the weights into both "
    "by 10",
    "igsp_y8_perfect": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and one intervention per pair of variables, each cutting both from their causes",
    "igsp_k_kci_k4": "additive nonlinear data from Erdos-Renyi graphs, observation and four "
    "interventions, kernel tests, outside the method's claim",
    "igsp_o1_perfect_k10": "linear-Gaussian data from Erdos-Renyi graphs with 50 and 100 "
    "variables, observation and ten interventions, outside the method's claim",
    "igsp_o2_perfect_k4": "linear-Gaussian data from Erdos-Renyi graphs with 10 variables, "
    "observation and four interventions, few rows each, outside the method's claim",
    "igsp_o2_perfect_k8": "linear-Gaussian data from Erdos-Renyi graphs with 20 variables, "
    "observation and eight interventions, few rows each, outside the method's claim",
    "igsp_o3_perfect_k8": "linear-Gaussian data from Erdos-Renyi graphs at mean degree 3, "
    "observation and eight interventions, outside the method's claim",
    "igsp_o8_wang_k4_laplace": "linear data with Laplace noise from Erdos-Renyi graphs, "
    "observation and four interventions, outside the method's claim",
    "igsp_o8_wang_k4_uniform": "linear data with uniform noise from Erdos-Renyi graphs, "
    "observation and four interventions, outside the method's claim",
    "igsp_o9_family45_c0p25": "linear-Gaussian data on the seven-variable graph of the IGSP paper, "
    "more observed rows than each intervention's, interventions on variables 4 and 5, outside the "
    "method's claim",
    "igsp_o9_family45_c0p25_c10": "linear-Gaussian data on the seven-variable graph of the IGSP "
    "paper, ten times as many observed rows as each intervention's, interventions on variables 4 "
    "and 5, outside the method's claim",
    "igsp_o9_family45_c0p25_c3": "linear-Gaussian data on the seven-variable graph of the IGSP "
    "paper, three times as many observed rows as each intervention's, interventions on variables 4 "
    "and 5, outside the method's claim",
    "igsp_o9_perfect_k4_c10": "linear-Gaussian data from Erdos-Renyi graphs, ten times as many "
    "observed rows as each of four interventions', outside the method's claim",
    "igsp_o9_perfect_k4_c3": "linear-Gaussian data from Erdos-Renyi graphs, three times as many "
    "observed rows as each of four interventions', outside the method's claim",
    "dynotears_paper": "linear time series with Gaussian noise, same-step and lag-1 links",
    "dynotears_paper_exponential": "linear time series with centered exponential noise",
    "dynotears_paper_ba_sbm": "linear time series from Barabasi-Albert and two-block graphs",
    "dynotears_paper_uncentered": "linear time series with exponential noise not centered, as in "
    "the paper's simulations",
    "dynotears_paper_standardized": "linear time series with Gaussian noise, standardized, outside "
    "the method's claim",
    "dynotears_sin_tanh": "time series with sine and tanh effects, outside the method's claim",
    "latent_anm_er": "additive-noise data from Erdos-Renyi graphs with one variable in five hidden",
    "latent_anm_er_tenth": "additive-noise data from Erdos-Renyi graphs with one variable in ten "
    "hidden",
    "camuv_paper": "nonlinear data from the CAM-UV paper's simulation, with hidden common causes "
    "and hidden intermediates",
    "fci_jci_s4": "linear-Gaussian data with hidden confounders from the JCI paper's simulation, "
    "observation and a mechanism change, 500 rows each",
    "fci_jci_s4p": "linear-Gaussian data with hidden confounders from the JCI paper's simulation, "
    "observation and a perfect intervention, 500 rows each",
    "fci_jci_l10": "linear-Gaussian data with hidden confounders from the JCI paper's simulation, "
    "observation and a mechanism change, 500 rows each",
    "fci_jci_x100": "linear-Gaussian data with hidden confounders from the JCI paper's simulation, "
    "observation and a mechanism change, 100 rows each",
    "fci_jci_s4m": "linear-Gaussian data with hidden confounders from the JCI paper's simulation, "
    "observation and two mechanism changes as one environment, outside the method's claim",
    "fci_jci_l10m": "linear-Gaussian data with hidden confounders from the JCI paper's simulation, "
    "observation and ten mechanism changes as one environment, outside the method's claim",
    "fci_jci_l10n100": "linear-Gaussian data with hidden confounders from the JCI paper's "
    "simulation, observation and a mechanism change, 100 rows each, outside the method's claim",
    "fci_jci_l10n2000": "linear-Gaussian data with hidden confounders from the JCI paper's "
    "simulation, observation and a mechanism change, 2,000 rows each, outside the method's claim",
    "bgf_p20_deg2_l0": "linear-Gaussian data from the BOSS-FCI paper's random DAGs at mean degree "
    "2, with no hidden variables",
    "bgf_p20_deg2_l4": "linear-Gaussian data from the BOSS-FCI paper's random DAGs at mean degree "
    "2, with up to 4 hidden confounders",
    "bgf_p20_deg2_l8": "linear-Gaussian data from the BOSS-FCI paper's random DAGs at mean degree "
    "2, with up to 8 hidden confounders",
    "bgf_p20_deg4_l0": "linear-Gaussian data from the BOSS-FCI paper's random DAGs at mean degree "
    "4, with no hidden variables",
    "bgf_p20_deg4_l4": "linear-Gaussian data from the BOSS-FCI paper's random DAGs at mean degree "
    "4, with up to 4 hidden confounders",
    "bgf_p20_deg4_l8": "linear-Gaussian data from the BOSS-FCI paper's random DAGs at mean degree "
    "4, with up to 8 hidden confounders",
    "bgf_p20_deg6_l0": "linear-Gaussian data from the BOSS-FCI paper's random DAGs at mean degree "
    "6, with no hidden variables",
    "bgf_p20_deg6_l4": "linear-Gaussian data from the BOSS-FCI paper's random DAGs at mean degree "
    "6, with up to 4 hidden confounders",
    "bgf_p20_deg6_l8": "linear-Gaussian data from the BOSS-FCI paper's random DAGs at mean degree "
    "6, with up to 8 hidden confounders",
    "bgf_o3_deg4_l4": "linear-Gaussian data from Erdos-Renyi graphs at mean degree 4 with 4 of 24 "
    "variables hidden, standardized, outside the method's claim",
}
# J-PCMCI+'s cells at alpha 0.05, keyed as its summary keys them: linear time series of 5 variables
# from the J-PCMCI+ paper's script, with every context observed, or one of each kind and the draws
# where no hidden context confounds an observed one and a variable; and, outside the claim, the
# model of andrey.data.sample_context_series and the script's model with 10 and 15 variables.
JPCMCIPLUS_PAPER = "linear time series from the J-PCMCI+ paper's script"
REGIMES |= {
    **{
        f"jpcmciplus_all_m10_t{t}": f"{JPCMCIPLUS_PAPER}, 10 datasets of {t} steps, every context "
        "observed"
        for t in (10, 20, 30, 50, 100, 150, 200)
    },
    **{
        f"jpcmciplus_half_m{m}_t{t}": f"{JPCMCIPLUS_PAPER}, {m} datasets of {t} steps, half the "
        "contexts observed"
        for m, t in [(10, t) for t in (10, 20, 30, 50, 100, 150, 200)]
        + [(m, 10) for m in (5, 20, 30, 50, 100, 150, 200)]
    },
    **{
        f"jpcmciplus_current_m{m}_t{t}": f"linear time series with same-step links, {m} datasets "
        f"of {t} steps, every context observed, outside the method's claim"
        for m, t in ((5, 100), (10, 50), (10, 100), (10, 200), (20, 50), (20, 100), (50, 10))
    },
    **{
        f"jpcmciplus_all_n{n}_m10_t100": f"{JPCMCIPLUS_PAPER} with {n} variables, 10 datasets of "
        "100 steps, every context observed, outside the method's claim"
        for n in (10, 15)
    },
}
# The 0.16.0 summary's cells: LOCI on cause-effect pairs whose noise scale changes with the cause,
# under Gaussian, Gumbel, and Laplace noise, each also with the cause rescaled to unit variance
# before the effect reads it, and on the five synthetic sets of Tagasovska et al. (2020).
LSNM_PAIRS = "cause-effect pairs with location-scale noise"
REGIMES |= {
    **{
        f"lsnm_{key}{tail}pair": f"{LSNM_PAIRS}, {noise}{scale}"
        for key, noise in (("", "Gaussian"), ("gumbel_", "Gumbel"), ("laplace_", "Laplace"))
        for tail, scale in (("", ""), ("rescaled_", ", the cause rescaled"))
    },
    **{
        f"tagasovska_{key}_pair": f"the {name} cause-effect pairs of Tagasovska et al. (2020)"
        for key, name in (
            ("an", "AN"),
            ("ans", "AN-s"),
            ("ls", "LS"),
            ("lss", "LS-s"),
            ("mnu", "MN-U"),
        )
    },
}
# The 0.17.0 summary's cells: SCORE and NoGAM on the simulation of their papers, with Gaussian,
# Laplace, or Gumbel noise, every column z-scored, or each variable rescaled to unit variance
# before its children read it.
STEIN_PAPER = "nonlinear data from the SCORE and NoGAM papers' simulation"
REGIMES |= {
    f"score_paper{key}{tail}": f"{STEIN_PAPER}, {noise} noise, {scale}"
    for key, noise in (("", "Gaussian"), ("_laplace", "Laplace"), ("_gumbel", "Gumbel"))
    for tail, scale in (("_standardized", "z-scored"), ("_rescaled", "each variable rescaled"))
}
# NOTEARS on the simulation of its paper, with Gaussian, exponential, or Gumbel noise and every
# column z-scored, the cells its claim is judged in.
REGIMES |= {
    f"notears_paper{key}_standardized": "linear data from the NOTEARS paper's simulation, "
    f"{noise} noise, z-scored"
    for key, noise in (("", "Gaussian"), ("_exponential", "exponential"), ("_gumbel", "Gumbel"))
}
# AGES on the 10-variable simulation of its paper, by the share of variable pairs that hold a weak
# edge. Its cells are at mean degrees from 0.9 to 9, so ``MEAN_DEGREE`` leaves them to the card.
REGIMES |= {
    f"ages_paper_weak{share}": "linear Gaussian data from the AGES paper's simulation, "
    + (f"{share}% of the variable pairs a weak edge" if share else "no weak edges")
    for share in (0, 10, 30, 50, 70)
}
METHOD_KEYS = {"method", "comparator", "regime", "hardware", "build", "rows"}
ROW_KEYS = {"d", "seeds", "andrey_s", "comparator_s", "ratio"}
# How each kind of summary counts a graph's errors: the row keys for Andrey's count, the other
# package's, and the empty graph's (None where a summary records none), then what one error is, as
# the accuracy section says it.
ERRORS = {
    "shd": (
        ("andrey_shd", "comparator_shd", "empty_shd"),
        "an edge to add, remove, or reorient to reach the true graph",
    ),
    "directed": (
        ("andrey_directed_errors", "comparator_directed_errors", "empty_directed_errors"),
        "on data with hidden common causes, a directed edge that only one of the learned and "
        "true graphs has",
    ),
    "direction": (
        ("andrey_reversed", "comparator_reversed", None),
        "on a cause-effect pair, the reversed direction",
    ),
    "temporal": (
        ("andrey_errors", "comparator_errors", "empty_errors"),
        "in a time series, a link that the learned and true graphs do not share, or a link end "
        "they mark differently",
    ),
    "latents": (
        ("andrey_latents_missed", "comparator_latents_missed", "empty_latents_missed"),
        "a latent variable whose indicators the learned model does not group as one cluster",
    ),
    "direct": (
        ("andrey_direct_errors_mean", "comparator_direct_errors_mean", "empty_direct_errors_mean"),
        "on data with hidden variables, an edge between two observed variables that no hidden "
        "variable joins, missing, added, or reversed, a reversed edge counting twice",
    ),
    # Last: the summaries of the other kinds give these keys too, beside their own.
    "window": (
        ("andrey_shd_mean", "comparator_shd_mean", "empty_shd_mean"),
        "in a time series, a same-step edge or a lag-1 link that the learned and true graphs do "
        "not share, or a same-step edge they orient differently",
    ),
}
# The density the homepage shows. A summary that measures more than one mean degree at a size
# gives the panel, the charts, and the counts its rows at mean degree 2, the density of the other
# summaries' Erdos-Renyi graphs.
MEAN_DEGREE = 2.0
# Comparisons excluded from page counts, keyed by the summary's ``comparison``: earlier Andrey
# builds and timings from a second machine that repeat the first machine's comparison.
NOT_COUNTED = {"earlier-build", "second-machine timing"}
# Regimes the homepage omits, as (lane, regime): additive data whose near-monotone effects
# partial correlation still detects, and the diabetes network, a single size with only a lower
# bound.
NOT_SHOWN = {("nonlinear", "anm_er"), ("boss-grasp", "diabetes")}
# Lanes the homepage omits: settings swept around a method's default, which its release card
# reports. IGSP at other levels and starts, UT-IGSP and FCI-JCI at their papers' levels, and
# BOSS-FCI and GRaSP-FCI at their paper's penalty and level. J-PCMCI+ shows its claimed level, the
# paper's 0.05, and omits the default 0.01, its context-link errors, the draws that break the
# paper's Assumption 3, and its nonlinear cells with GPDC, whose reference reads one column of the
# dummies. LOCI shows its default criterion, independence, and omits the likelihood criterion.
# SCORE and NoGAM show their z-scored and rescaled cells and omit the cells in the simulation's
# own units, where the references read those units and Andrey standardizes. NOTEARS omits its
# cells in the simulation's own units and its rescaled cells, which its release card reports and
# does not judge; its z-scored cells are at mean degrees 4 and 8, the paper's, so ``MEAN_DEGREE``
# leaves them to the card too.
NOT_SHOWN_LANES = {
    "igsp.alpha_1e-2_starts_5",
    "igsp.alpha_1e-2_starts_10",
    "igsp.alpha_1e-3_starts_5",
    "igsp.alpha_1e-4_starts_5",
    "igsp.alpha_1e-4_starts_10",
    "ut_igsp.paper_level",
    "fci_jci.alpha_0.01",
    "boss_grasp_fci.paper",
    "jpcmciplus.alpha_0.01",
    "jpcmciplus.context_links",
    "jpcmciplus.assumption_3_broken",
    "jpcmciplus.gpdc",
    "loci.likelihood",
    "score_matching.raw",
    "nogam.raw",
    "notears.raw",
    "notears.rescaled",
}
# Packages the homepage omits, each reported on its release card beside the default comparison: a
# reference patched to follow its paper where the package does not; pcalg under the JCI
# assumptions that match FCI-JCI's non-default confounded_index=False; an earlier Tetrad release,
# 7.6.3 in causal-cmd 1.12.0, measured as context; Tetrad 7.6.3 against the
# edge_removal="triangles" option of BOSS-FCI and GRaSP-FCI; and lingam's CAM-UV at its default
# set size of 2, beside the paper's size of 3, Andrey's default.
NOT_SHOWN_COMPARATORS = {
    "graphical-model-learning 0.1a8, patched to Definition 4.2",
    'pcalg 2.7-12, jci = "123"',
    "causal-cmd 1.12.0 bfci (tetrad-lib 7.6.3)",
    "causal-cmd 1.12.0 grasp-fci (tetrad-lib 7.6.3)",
    "Tetrad 7.6.3",
    "lingam 1.13.0 at its default set size",
}
# The package each comparator names, without the version or setting its release card gives;
# every other comparator is a package's name already.
PACKAGE_NAMES = {
    "CAM 1.0": "CAM",
    "InvariantCausalPrediction 0.8": "InvariantCausalPrediction",
    "causal-learn 0.1.4.8": "causal-learn",
    "lingam 1.13.0": "lingam",
    "lingam RESIT": "lingam",
    'pcalg 2.7-12, jci = "1"': "pcalg",
    "tetrad-lib 7.6.10 BossFci": "Tetrad",
    "tetrad-lib 7.6.10 GraspFci": "Tetrad",
    "tigramite 5.2.10.1": "tigramite",
    "gnies 0.3.3": "gnies",
    "CausalNex 0.12.2": "CausalNex",
    "graphical-model-learning 0.1a8": "graphical-model-learning",
    "LOCI authors' code, LOCI-H": "LOCI authors' code",
}
# The summary lanes that measure each graph size at several sample sizes.
GRID_LANES = {"nonlinear", "boss_grasp_fci"}
# A method's line uses its first shortest-named comparison by default. These settings override that
# choice, each shown under the method's name from its first comparison: BOSS and GRaSP with the
# BDeu score on categorical data, as measured in 0.8.0, and ICP with its default approximate test.
SHOWN_SETTINGS = {
    "BOSS on discrete data, BDeu score": "BOSS",
    "GRaSP on discrete data, BDeu score": "GRaSP",
    "ICP, approximate test": "ICP",
}
DEMO_URL: str | None = None  # the live demo; link it once its Space is public
# The one-line description, as pyproject.toml and the README give it.
DESCRIPTION = "Andrey: accelerated, accurate, agentic causal discovery."
# The homepage's search and link-preview title includes "fast", "causal discovery", "Python",
# and "agents" alongside the product name.
SEARCH_TITLE = "Andrey: fast causal discovery in Python, built for AI agents"

# Graph colors resolve through the page's tokens, so the drawing follows the light or dark theme.
TOKEN_PALETTE = andrey.viz.Palette(
    **{
        field.name: (
            getattr(andrey.viz.LIGHT, field.name)
            if isinstance(getattr(andrey.viz.LIGHT, field.name), tuple)
            else f"var(--andrey-{field.name.replace('_', '-')})"
        )
        for field in dataclasses.fields(andrey.viz.LIGHT)
    }
)


# ---- the summaries ---------------------------------------------------------------------------
def load_summary(path: Path, *, placeholder: bool = False) -> dict:
    """Read a release's benchmark summary and check its layout; refuse unpublished files for
    release. Comparisons in ``NOT_COUNTED``, regimes in ``NOT_SHOWN``, lanes in
    ``NOT_SHOWN_LANES``, and packages in ``NOT_SHOWN_COMPARATORS`` are excluded."""
    if not placeholder and not path.resolve().is_relative_to(PUBLISHED):
        raise ValueError(
            f"{path} is not under benchmarks/published/; preview it with -D andrey_placeholder=1"
        )
    if not path.exists():
        raise ValueError(f"{path} does not exist; it ships in benchmarks/published/")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("placeholder") and not placeholder:
        raise ValueError(
            f"{path} holds placeholder numbers; preview it with -D andrey_placeholder=1"
        )
    missing = {"generated", "methods"} - summary.keys()
    if missing:
        raise ValueError(f"summary lacks {sorted(missing)}")
    # Earlier-build comparisons measure Andrey's progress; second-machine timings repeat a
    # comparison. Both appear on the release card and are excluded from page counts.
    # Regimes in ``NOT_SHOWN``, lanes in ``NOT_SHOWN_LANES``, and packages in
    # ``NOT_SHOWN_COMPARATORS`` are also excluded.
    summary = {
        **summary,
        "methods": [
            m
            for m in summary["methods"]
            if m.get("comparison") not in NOT_COUNTED
            and (m.get("lane"), m.get("regime")) not in NOT_SHOWN
            and m.get("lane") not in NOT_SHOWN_LANES
            and m.get("comparator") not in NOT_SHOWN_COMPARATORS
        ],
    }
    for method in summary["methods"]:
        name = method.get("method", "?")
        if METHOD_KEYS - method.keys():
            raise ValueError(f"{name} lacks {sorted(METHOD_KEYS - method.keys())}")
        if method["regime"] not in REGIMES:
            raise ValueError(f"{name}: describe regime {method['regime']!r} in REGIMES")
        registry_name(method)
        kind = error_kind(method)
        keys = ROW_KEYS | {key for key in ERRORS[kind][0] if key}
        for row in method["rows"]:
            if keys - row.keys():
                raise ValueError(f"{name} d={row.get('d')} lacks {sorted(keys - row.keys())}")
            if kind == "direction" and (row.get("andrey_no_edge") or row.get("comparator_no_edge")):
                raise ValueError(f"{name} n={row['n']}: a pair without an edge is not counted")
            timed = row["andrey_s"] is not None and row["comparator_s"] is not None
            if row["ratio"] is not None and not timed:
                raise ValueError(
                    f"{name} d={row['d']}: a ratio is published, but a time is missing"
                )
            if row.get("ratio_is_lower_bound") and row.get("ratio_lower_bound") is None:
                raise ValueError(f"{name} d={row['d']}: a lower bound is flagged, but not given")
    return summary


def release_summaries(summary: dict) -> list[dict]:
    """Every release's summary, in ``RELEASES`` order, with ``summary`` standing in for the
    first, each comparison only in the newest summary that measures it (:func:`newest`)."""
    return newest([summary, *(load_summary(PUBLISHED / f / "summary.json") for f in RELEASES[1:])])


def newest(summaries: list[dict]) -> list[dict]:
    """``summaries`` with each comparison kept only in the newest summary that measures it.

    A comparison is a summary's method, by its name (``method``), against one package
    (``comparator``), so a method measured under two settings is two comparisons; this is the key
    ``qa/reference_parity.py`` judges by. A released summary never changes, so when a later summary
    measures the same comparison, the page shows the later rows and drops the earlier summary's.
    "Later" is the summary's ``generated`` date; two summaries generated on the same day that both
    measure a comparison raise. The list keeps its order and length, so it still lines up with
    ``RELEASES``.
    """
    dated: dict[tuple[str, str], str] = {}
    for summary in summaries:
        for key in {(m["method"], m["comparator"]) for m in summary["methods"]}:
            seen = dated.get(key)
            if seen == summary["generated"]:
                raise ValueError(f"{key} is in two summaries generated on {seen}")
            if seen is None or seen < summary["generated"]:
                dated[key] = summary["generated"]
    return [
        {
            **summary,
            "methods": [
                m
                for m in summary["methods"]
                if dated[(m["method"], m["comparator"])] == summary["generated"]
            ],
        }
        for summary in summaries
    ]


def normalized(name: str) -> str:
    """Match a summary method ("DirectLiNGAM") to a registry name ("direct_lingam")."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def registry_name(method: dict) -> str:
    """The registry's name for a summary's method: the method its ``solution`` runs
    (``andrey.tabu.numpy.serial`` is ``tabu``), or else its own name matched to the registry's
    ("DirectLiNGAM" is ``direct_lingam``)."""
    names = {s.name for s in list_specs()}
    if "solution" in method:
        name = method["solution"].split(".")[1]
    else:
        name = next((n for n in names if normalized(n) == normalized(method["method"])), "")
    if name not in names:
        raise ValueError(f"{method['method']}: no registered method runs it")
    return name


def error_kind(method: dict) -> str:
    """How a comparison counts a graph's errors: the first kind in ``ERRORS`` whose keys its rows
    give.

    The accuracy comparison sets Andrey's count against the other package's on the same datasets
    at each size, in the measure the summary records for both:

    - ``shd``: the structural Hamming distance to the true graph, the edges to add, remove, or
      reorient to reach it (the 0.1.0 and 0.3.0 summaries).
    - ``directed``: on data with hidden common causes, the directed edges in one graph but not the
      other, against the true graph's projection onto the observed variables; a reversed edge
      counts twice (the 0.2.0 summary).
    - ``direction``: on cause-effect pairs, the pairs whose direction is reversed (the
      0.4.0 summary). Each package names a direction on every pair it fits, so a fitted
      pair is either right or reversed; a summary with a pair left without an edge fails to load.
    - ``temporal``: in a time series, the errors in the method's own unit, which its summary's
      ``error_metric`` defines: the lag-resolved SHD, the endpoint SHD over a window of lags, or
      the lagged or cross-link edges missed plus those added (the 0.5.0 summary).
    - ``latents``: for a method that finds latent variables from their indicators, the true
      latents whose indicators it does not return as one cluster (GIN in the 0.7.0 summary).
    - ``direct``: on data with hidden variables, the edges between observed variables that no
      hidden variable joins, missing or added, a reversed edge counting twice; means over the
      paired datasets (CAM-UV in the 0.14.0 summary).
    - ``window``: in a time series, the SHD over lags 0 and 1, a same-step pair whose edge differs
      or a lag-1 link in one graph and not the other; means over the paired series (DYNOTEARS in
      the 0.14.0 summary). Other summaries give these keys beside their own kind's, so it comes
      last.

    Fewer is better in each. A comparison whose rows give none of them fails, rather than drop out
    of the count.
    """
    kinds = [
        kind
        for kind, ((ours, _, _), _) in ERRORS.items()
        if all(ours in row for row in method["rows"])
    ]
    if not kinds:
        raise ValueError(f"{method.get('method', '?')}: name how its rows count errors in ERRORS")
    return kinds[0]


def claimed_ratio(row: dict) -> tuple[float | None, bool]:
    """A row's speedup and whether it is a lower bound. Where a package hit its time cap on some
    datasets, the summary flags ``ratio_is_lower_bound`` and gives the bound as
    ``ratio_lower_bound``, which stands for ``ratio``."""
    if row.get("ratio_is_lower_bound"):
        return row["ratio_lower_bound"], True
    return row["ratio"], False


def cpu_type(row: dict) -> str | None:
    """The CPU used for a row's timings, from ``hardware`` before the core count."""
    return (row.get("hardware") or "").split(",")[0] or None


def shown_rows(method: dict) -> list[dict]:
    """A comparison's rows selected for the homepage, along one line of graph sizes.

    - Rows at a mean degree other than ``MEAN_DEGREE`` are left out.
    - A lane in ``GRID_LANES`` measures each graph size at several sample sizes. Only rows at the
      smallest sample size with a recorded speedup remain.
    - If timings cover several CPU types, only the CPU type with the lowest minimum speedup
      remains, along with rows that name no CPU.
    """
    rows = [r for r in method["rows"] if r.get("mean_degree", MEAN_DEGREE) == MEAN_DEGREE]
    timed = [r for r in rows if claimed_ratio(r)[0] is not None]
    if method.get("lane") in GRID_LANES and timed:
        smallest = min(r["n"] for r in timed)
        rows = [r for r in rows if r["n"] == smallest]
        timed = [r for r in timed if r["n"] == smallest]
    cpus = {cpu_type(r) for r in timed} - {None}
    if len(cpus) > 1:
        slowest = min(
            cpus, key=lambda c: min(claimed_ratio(r)[0] for r in timed if cpu_type(r) == c)
        )
        rows = [r for r in rows if cpu_type(r) in (slowest, None)]
    return rows


def comparisons(summary: dict) -> list[dict]:
    """A summary's comparisons in one layout, whatever kind of summary it is.

    Each keeps the summary's method, package (``comparator``), regime, hardware, and build, and
    adds the registry's name (``spec``), family, and status, its ``error_kind``, and the ``unit``
    of its sizes: "variables", or "rows" for a pairwise method. Each row gives:

    - ``d``: the size, in ``unit``: the variables the method gets, the observed ones on data with
      hidden variables, or a cause-effect pair's rows, since every pair has two variables;
    - ``n``: the sample size, ``None`` where the summary records none;
    - ``andrey_s`` and ``comparator_s``: the median fit times, ``None`` past a time cap;
    - ``ratio`` and ``at_least``: the speedup, from :func:`claimed_ratio`;
    - ``andrey_errors`` and ``comparator_errors``: each graph's errors, from :func:`error_kind`;
    - ``card_verdict``: the release card's verdict on the row, "better", "tie", or "worse"; null
      where the card did not judge it, and "unrecorded" where the summary predates the field.

    Rows are those :func:`shown_rows` keeps. The summary's ``reach`` rows are excluded: they cover
    sizes the other package did not finish or run, so they have no paired time.
    """
    specs = {s.name: s for s in list_specs()}
    out = []
    for method in summary["methods"]:
        spec = specs[registry_name(method)]
        kind = error_kind(method)
        ours, theirs, empty = ERRORS[kind][0]
        pairs = spec.family == "pairwise"
        rows = []
        for row in shown_rows(method):
            ratio, at_least = claimed_ratio(row)
            rows.append(
                {
                    "d": row["n"] if pairs else row.get("observed", row["d"]),
                    "n": row.get("n"),
                    "andrey_s": row["andrey_s"],
                    "comparator_s": row["comparator_s"],
                    "ratio": ratio,
                    "at_least": at_least,
                    "andrey_errors": row[ours],
                    "comparator_errors": row[theirs],
                    "card_verdict": row.get("card_verdict", "unrecorded"),
                    # The empty graph's error count; in pairwise comparisons, it gets every
                    # pair wrong.
                    "empty_errors": row["paired"] if pairs else row.get(empty) if empty else None,
                    # A temporal summary's link F1: cross-links, or LPCMCI's adjacencies.
                    "andrey_f1": row.get("andrey_cross_f1", row.get("andrey_adjacency_f1")),
                    "comparator_f1": row.get(
                        "comparator_cross_f1", row.get("comparator_adjacency_f1")
                    ),
                }
            )
        out.append(
            {
                **method,
                "spec": spec.name,
                "release": summary.get("release", RELEASES[0]),
                "family": spec.family,
                "status": spec.status,
                "error_kind": kind,
                "unit": "rows" if pairs else "variables",
                "rows": rows,
            }
        )
    return out


def supported(summaries: list[dict]) -> list[dict]:
    """Every comparison in ``summaries`` whose method the registry supports, grouped by family in
    ``SPEC_FAMILIES`` order, each family's in the summaries' order. Only these count on the
    page."""
    rank = {family: i for i, family in enumerate(SPEC_FAMILIES)}
    found = [m for s in summaries for m in comparisons(s) if m["status"] == "supported"]
    return sorted(found, key=lambda method: rank[method["family"]])


def sizes_text(sizes: list[int]) -> str:
    """Join problem sizes: d = 400; d = 200 and 400; d = 20, 50, and 100."""
    words = [f"{d:,}" for d in sizes]
    if len(words) > 2:
        words = [", ".join(words[:-1]) + ",", words[-1]]
    return "d = " + " and ".join(words)


def listed(words: list[str]) -> str:
    """Join words as a list: a; a and b; a, b, and c."""
    if len(words) > 2:
        return ", ".join(words[:-1]) + f", and {words[-1]}"
    return " and ".join(words)


def ordered(summary: dict) -> list[dict]:
    """A summary's comparisons by registry family, in ``SPEC_FAMILIES`` order, keeping the
    summary's order within one."""
    rank = {family: i for i, family in enumerate(SPEC_FAMILIES)}
    family = {s.name: s.family for s in list_specs()}
    return sorted(summary["methods"], key=lambda m: rank[family[registry_name(m)]])


def _largest(found: list[dict]) -> tuple[dict, dict]:
    """The comparison and row with the largest ratio."""
    cells = [(m, r) for m in found for r in m["rows"] if r["ratio"] is not None]
    method, row = max(cells, key=lambda cell: cell[1]["ratio"])
    if row["ratio"] <= 1:
        raise ValueError("no comparison shows Andrey faster")
    return method, row


def faster(method: dict) -> bool:
    """Whether Andrey is faster in a comparison at the largest size with a published ratio."""
    rows = [r for r in method["rows"] if r["ratio"] is not None]
    return bool(rows) and max(rows, key=lambda r: r["d"])["ratio"] > 1


def speed_claim(summaries: list[dict]) -> tuple[str, str]:
    """The speed claim over every supported comparison, in two parts, with no single ratio a rerun
    would move.

    The lead rounds the largest ratio down to a power of ten and names its method ("Over 100x
    faster on PC"). The rest says how many of the other supported methods with a published ratio
    are faster against every package they were compared with: "the other" when all are, "most
    other" when more than half are, and nothing when fewer are. A method is the registry's, so
    every comparison of one method, with any of its settings, has to be faster.
    """
    found = supported(summaries)
    method, row = _largest(found)
    floor = 10 ** math.floor(math.log10(row["ratio"]))
    lead = f"{'Over' if row['ratio'] > floor else 'At least'} {ratio_text(floor)} faster on "
    lead += method["method"]
    others: dict[str, bool] = {}
    for m in found:
        if m["spec"] != method["spec"] and any(r["ratio"] is not None for r in m["rows"]):
            others[m["spec"]] = others.get(m["spec"], True) and faster(m)
    count = sum(others.values())
    if count == len(others):
        rest = "faster on the other supported methods"
    elif 2 * count > len(others):
        rest = "faster on most other supported methods"
    else:
        rest = ""
    return lead, rest


# The speed claim the README, the live demo, and the homepage's description state.
CLAIM = "The fastest causal discovery package."


def claim_sentence(summaries: list[dict]) -> str:
    """The speed claim as one sentence, the form of a summary's ``headline``."""
    lead, rest = speed_claim(summaries)
    return f"{lead}, and {rest}." if rest else f"{lead}."


def fastest(summary: dict) -> tuple[dict, dict]:
    """Return the comparison and row with the largest ratio in one summary; its headline must be
    the claim its numbers support."""
    method, row = _largest(supported([summary]))
    if summary.get("headline") != claim_sentence([summary]):
        raise ValueError(
            f"headline {summary.get('headline')!r} is not the claim the numbers support, "
            f"{claim_sentence([summary])!r} ({method['method']} at d = {row['d']}: "
            f"{ratio_text(row['ratio'])})"
        )
    return method, row


def claimed_comparisons(summaries: list[dict]) -> list[dict]:
    """The comparisons the charts cover: the supported ones with a published ratio at some size."""
    return [m for m in supported(summaries) if any(r["ratio"] is not None for r in m["rows"])]


def time_groups(summaries: list[dict]) -> list[tuple[str, list[dict]]]:
    """The comparisons a chart each: a method's comparisons share one chart when they come from
    the same runs, otherwise one chart apiece. A comparison joins a chart when it measures a size
    every comparison in the chart measures, and Andrey's times agree with each of them at every
    size they share."""

    def times(method: dict) -> dict[int, Counter]:
        out: dict[int, Counter] = {}
        for r in method["rows"]:
            out.setdefault(r["d"], Counter())[r["andrey_s"]] += 1
        return out

    groups: list[tuple[str, list[dict]]] = []
    for method in claimed_comparisons(summaries):
        ours = times(method)
        same = [
            g
            for g in groups
            if g[0] == method["method"]
            and set(ours).intersection(*(times(m).keys() for m in g[1]))
            and all(
                theirs[d] == ours[d]
                for theirs in map(times, g[1])
                for d in theirs.keys() & ours.keys()
            )
        ]
        if same:
            same[0][1].append(method)
        else:
            groups.append((method["method"], [method]))
    return groups


def andrey_rows(methods: list[dict]) -> list[dict]:
    """Andrey's rows across a chart's comparisons (:func:`time_groups`), smallest size first: each
    size from the first comparison that measures it, since the chart's comparisons agree there."""
    rows: list[dict] = []
    for method in methods:
        known = {r["d"] for r in rows}
        rows += [r for r in method["rows"] if r["d"] not in known]
    return sorted(rows, key=lambda r: r["d"])


def log_scale(lo: float, hi: float, a: float, b: float):
    """Map ``lo..hi`` on a log scale onto ``a..b``."""
    span = math.log10(hi / lo)
    return lambda v: a + (b - a) * math.log10(v / lo) / span


def time_chart(name: str, methods: list[dict], xr: tuple[float, float]) -> str:
    """Fit time (x) against problem size (y) for Andrey and each package, both on log scales.

    Up and to the left is better: larger problems in less time. The size is in the comparisons'
    ``unit``, variables or rows, and its axis spans and labels the sizes the chart measures. A
    hollow point is a size with no paired ratio; a package past its time cap has no point at that
    size.
    """
    unit = methods[0]["unit"]
    sizes = sorted({r["d"] for m in methods for r in m["rows"]})
    w, h, left, right, top, bottom = 320, 232, 44, 306, 22, 196
    x = log_scale(xr[0], xr[1], left, right)
    lo, hi = (sizes[0], sizes[-1]) if len(sizes) > 1 else (sizes[0] / 2, sizes[0] * 2)
    y = log_scale(lo, hi, bottom, top)
    grid = []
    for k in range(round(math.log10(xr[0])), round(math.log10(xr[1])) + 1):
        v = 10.0**k
        grid.append(
            f'<line x1="{x(v):.1f}" x2="{x(v):.1f}" y1="{top}" y2="{bottom}" class="tc-grid"/>'
            f'<text x="{x(v):.1f}" y="{bottom + 14}" text-anchor="middle">{seconds_axis(v)}</text>'
        )
    for d in sizes:
        grid.append(
            f'<line x1="{left}" x2="{right}" y1="{y(d):.1f}" y2="{y(d):.1f}" class="tc-grid"/>'
            f'<text x="{left - 6}" y="{y(d) + 3:.1f}" text-anchor="end">{d:,}</text>'
        )
    ours = [(r["d"], r["andrey_s"], r["ratio"] is not None) for r in andrey_rows(methods)]
    series = [("Andrey", ours, "tc-andrey")]
    series += [
        (
            m["comparator"],
            [(r["d"], r["comparator_s"], r["ratio"] is not None) for r in m["rows"]],
            f"tc-other tc-other-{i}",
        )
        for i, m in enumerate(methods)
    ]
    marks = []
    for label, points, cls in series:
        timed = [(d, t, claim) for d, t, claim in points if t is not None]
        if not timed:
            continue
        path = " ".join(f"{x(t):.1f},{y(d):.1f}" for d, t, _ in timed)
        marks.append(f'<polyline class="tc-line {cls}" points="{path}"/>')
        for d, t, claim in timed:
            tip = f"{label}, {d:,} {unit}: {seconds_text(t)}" + (
                "" if claim else ", no paired ratio"
            )
            marks.append(
                f'<circle class="tc-point {cls}{"" if claim else " is-open"}" cx="{x(t):.1f}" '
                f'cy="{y(d):.1f}" r="4"><title>{escape(tip)}</title></circle>'
            )
    best = max(
        (r for m in methods for r in m["rows"] if r["ratio"] is not None),
        key=lambda r: r["ratio"],
    )
    _, shown = more_than(best["ratio"])
    peers = " and ".join(m["comparator"] for m in methods)
    legend = "".join(
        f'<span class="tc-key {cls}"><i aria-hidden="true"></i>{escape(label)}</span>'
        for label, _, cls in series
    )
    return (
        '<figure class="tchart"><figcaption>'
        f'<span class="tchart-name"><strong>{escape(name)}</strong></span>'
        f'<span class="tchart-up"><b>{shown:,}x+</b></span></figcaption>'
        f'<p class="tc-legend">{legend}</p>'
        f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(name)}: fit time against '
        f'the number of {unit} for Andrey and {escape(peers)}">'
        f'<g class="tc-axis">{"".join(grid)}</g>'
        f'<text class="tc-title" x="{left}" y="{h - 4}">seconds (log)</text>'
        f'<text class="tc-title" x="4" y="{top - 8}">{unit}</text>'
        f"{''.join(marks)}</svg></figure>"
    )


def seconds_axis(v: float) -> str:
    """A decade of seconds as an axis label: 0.01, 0.1, 1, 10, 100, 1k, 10k."""
    return {1000: "1k", 10000: "10k"}.get(v, f"{v:g}")


def speed_groups(summaries: list[dict]) -> list[tuple[str, list[dict]]]:
    """Groups from ``time_groups`` where Andrey is faster in every comparison.

    Other comparisons appear in the release cards' tables.
    """
    return [(n, ms) for n, ms in time_groups(summaries) if all(faster(m) for m in ms)]


def speed_figures(summaries: list[dict]) -> list[tuple[str, str]]:
    """A chart per method (or per run where a method's comparisons differ), each with its largest
    speedup, as (registry family, chart). The charts share one time axis, so they compare at a
    glance; each size axis spans its own chart's sizes."""
    groups = speed_groups(summaries)
    times = [
        t
        for _, methods in groups
        for m in methods
        for r in m["rows"]
        for t in (r["andrey_s"], r["comparator_s"])
        if t is not None
    ]
    xr = (10 ** math.floor(math.log10(min(times))), 10 ** math.ceil(math.log10(max(times))))
    return [(methods[0]["family"], time_chart(name, methods, xr)) for name, methods in groups]


def speed_chart(summaries: list[dict]) -> str:
    """The speed charts on a belt, for the benchmark page and the live demo."""
    return belt([figure for _, figure in speed_figures(summaries)])


def belt(figures: list[str]) -> str:
    """Figures on a belt that slides sideways in a loop. The second copy makes the loop seamless
    and is hidden from assistive technology; the duration grows with the count, so the speed
    holds."""
    shown = "".join(figures)
    copy = re.sub(r'<figure class="([^"]*)">', r'<figure class="\1" aria-hidden="true">', shown)
    return (
        f'<div class="tc-belt"><div class="tc-track" style="--n: {len(figures)}">'
        f'{shown}<div class="tc-copy" aria-hidden="true">{copy}</div></div></div>'
    )


def exceptions(summary: dict) -> list[str]:
    """Every cell where Andrey is slower, has no paired ratio, or a package ran past its cap."""
    notes = []
    for method in ordered(summary):
        name, comparator, rows = method["method"], method["comparator"], method["rows"]
        timed = [r for r in rows if r["andrey_s"] is not None and r["comparator_s"] is not None]
        if timed and all(r["ratio"] is None for r in rows):
            # The charts skip this comparison, and the table below lists its rows, so the notes
            # say nothing about it beyond a run that hit its cap.
            notes += capped_notes(name, comparator, rows)
            continue
        for r in rows:
            if r["ratio"] is not None and r["ratio"] < 1:
                notes.append(
                    f"{name} is slower than {comparator} at d = {r['d']:,}: "
                    f"{ratio_text(r['ratio'])}, {seconds_text(r['andrey_s'])} compared with "
                    f"{seconds_text(r['comparator_s'])}."
                )
        unpaired = [r["d"] for r in timed if r["ratio"] is None]
        if unpaired:
            notes.append(
                f"No paired datasets for {name} compared with {comparator} at "
                f"{sizes_text(unpaired)}, so no ratio."
            )
        notes += capped_notes(name, comparator, rows)
    return notes


def capped_notes(name: str, comparator: str, rows: list[dict]) -> list[str]:
    """Name the sizes where a package ran past its time cap."""
    notes = []
    for side, key in ((comparator, "comparator_s"), ("Andrey", "andrey_s")):
        capped = [r["d"] for r in rows if r[key] is None]
        if capped:
            notes.append(
                f"{side} did not finish {name} within the time cap at {sizes_text(capped)}."
            )
    return notes


# ---- quality in the same comparisons ---------------------------------------------------------
def error_comparisons(methods: list[dict]) -> list[dict]:
    """Each size where both packages have a graph, per package: Andrey's errors against theirs.

    The errors are each comparison's own count (:func:`error_kind`). The change is a share of the
    other package's errors, rounded against Andrey: fewer rounds down and more rounds up. Against
    a package with no errors it is a count. Past a time cap a package has no graph to count.
    """
    out = []
    for d in sorted({r["d"] for m in methods for r in m["rows"]}):
        for method in methods:
            row = next((r for r in method["rows"] if r["d"] == d), None)
            if row is None or row["andrey_errors"] is None or row["comparator_errors"] is None:
                continue
            ours, theirs = row["andrey_errors"], row["comparator_errors"]
            if abs(ours - theirs) < 1e-9:
                kind, change, label = "same", 0.0, "same"
            elif theirs == 0:
                kind, change, label = "more", 100.0, f"{number_text(ours)} more"
            else:
                change = 100 * abs(ours - theirs) / theirs
                kind = "fewer" if ours < theirs else "more"
                whole = math.floor(change) if kind == "fewer" else math.ceil(change)
                label = f"{whole}% {kind}" if whole else f"<1% {kind}"
            out.append(
                {
                    "d": d,
                    "method": method,
                    "row": row,
                    "kind": kind,
                    "change": change,
                    "label": label,
                }
            )
    return out


def settings_text(cells: list[tuple[dict, dict]]) -> str:
    """Name settings, a comparison and row each, with one comparison's sizes together: "FCI
    compared with causal-learn on <regime> at 50 and 100 variables"."""
    sizes: dict[tuple[str, str, str, str], list[int]] = {}
    for method, row in cells:
        key = (method["method"], method["comparator"], method["regime"], method["unit"])
        sizes.setdefault(key, []).append(row["d"])
    return listed(
        [
            f"{name} compared with {comparator} on {REGIMES[regime]} at "
            f"{listed([f'{d:,}' for d in dict.fromkeys(ds)])} {unit}"
            for (name, comparator, regime, unit), ds in sizes.items()
        ]
    )


def recorded_outcomes(summaries: list[dict]) -> tuple[dict[str, list], dict[str, list]]:
    """The outcome the summaries record at each setting of the comparisons the charts cover
    (:func:`claimed_comparisons`), as (comparison, row) pairs, in two groups.

    The first holds the settings a release card judges (``card_verdict``), by verdict: "tie"
    where the paired 95% interval of the difference in errors contains 0, "better", or "worse";
    a setting the card did not judge is left out. The second holds the settings from summaries
    that predate ``card_verdict``, by the two packages' error counts alone, as
    :func:`error_comparisons` compares them: "same", "fewer", or "more". Past a time cap a
    package has no graph to count.
    """
    judged: dict[str, list] = {"tie": [], "better": [], "worse": []}
    counted: dict[str, list] = {"same": [], "fewer": [], "more": []}
    for m in claimed_comparisons(summaries):
        for r in m["rows"]:
            ours, theirs = r["andrey_errors"], r["comparator_errors"]
            if r["card_verdict"] in judged:
                judged[r["card_verdict"]].append((m, r))
            elif r["card_verdict"] == "unrecorded" and ours is not None and theirs is not None:
                kind = "same" if abs(ours - theirs) < 1e-9 else "fewer" if ours < theirs else "more"
                counted[kind].append((m, r))
    return judged, counted


def quality_intro(summaries: list[dict]) -> str:
    """Say what an error is in each count the comparisons use, the outcome the summaries record
    (:func:`recorded_outcomes`), naming every setting where Andrey makes more errors, and that
    releases are tested against true graphs (the quality gate, which every publish job needs)."""
    kinds = dict.fromkeys(m["error_kind"] for m in claimed_comparisons(summaries))
    errors = ", or, ".join(ERRORS[kind][1] for kind in kinds)
    judged, counted = recorded_outcomes(summaries)

    def outcome(cells: list) -> str:
        return str(len(cells)) if cells else "none"

    def named(cells: list) -> str:
        return f": {settings_text(cells)}." if cells else "."

    text = f"An error is {errors}. The charts give both packages' errors at each size."
    if any(judged.values()):
        text += (
            f" Of the {sum(map(len, judged.values()))} settings in the charted comparisons that "
            "a release card judges, Andrey ties the other package in "
            f"{outcome(judged['tie'])}, where the paired 95% interval of the difference in errors "
            f"contains 0, makes fewer errors in {outcome(judged['better'])}, and makes more in "
            f"{outcome(judged['worse'])}" + named(judged["worse"])
        )
    if any(counted.values()):
        text += (
            f" Of the {sum(map(len, counted.values()))} settings in the charted comparisons "
            "without a card's verdict, compared by their error counts alone, Andrey makes the "
            f"same number of errors as the other package in {outcome(counted['same'])}, fewer "
            f"in {outcome(counted['fewer'])}, and more in {outcome(counted['more'])}"
            + named(counted["more"])
        )
    return text + " Every release is tested against the true graphs that generated the data."


def accuracy_card(name: str, methods: list[dict]) -> str:
    """One method's comparisons as a table: each size, each package, and Andrey's errors against
    theirs. Every card has the same columns, so rows line up across cards; the bar grows with the
    share of errors Andrey avoids."""
    rows, first_at = [], set()
    scored = error_comparisons(methods)
    for c in scored:
        method, row, d = c["method"], c["row"], c["d"]
        size = "" if d in first_at else f"{d:,}"
        first_at.add(d)
        if c["kind"] == "fewer":
            mark = f'<span class="rt-bar" style="--f: {min(c["change"], 100):.1f}%"></span>'
        elif c["kind"] == "same":
            mark = '<span class="rt-dot"></span>'
        else:
            mark = ""
        tip = (
            f"{method['comparator']}, {d:,} {method['unit']}: Andrey "
            f"{number_text(row['andrey_errors'])} errors, {method['comparator']} "
            f"{number_text(row['comparator_errors'])}"
        )
        rows.append(
            f'<span class="rt-d">{size}</span><span class="rt-who">'
            f"{escape(method['comparator'])}</span>"
            f'<span class="rt-track" title="{escape(tip)}">{mark}</span>'
            f'<span class="rt-v">{escape(c["label"])}</span>'
        )
    good = sum(c["kind"] != "more" for c in scored)
    return (
        '<figure class="tchart rtable"><figcaption>'
        f'<strong class="rt-name">{escape(name)}</strong>'
        f'<span class="rt-score"><b>{good} of {len(scored)}</b>'
        "<span>same or fewer errors</span></span></figcaption>"
        f'<div class="rt-grid"><span class="rt-h rt-d">{methods[0]["unit"].capitalize()}</span>'
        '<span class="rt-h">Compared with</span><span class="rt-h">Fewer errors &rarr;</span>'
        f'<span class="rt-h"></span>{"".join(rows)}</div></figure>'
    )


def accuracy_cards(summaries: list[dict]) -> list[str]:
    """A card per speed chart, for the comparisons with a graph each."""
    groups = [(n, ms) for n, ms in time_groups(summaries) if error_comparisons(ms)]
    return [accuracy_card(name, methods) for name, methods in groups]


def accuracy_rows(summaries: list[dict]) -> list[dict]:
    """Per method and package compared: the method's name and spec, the package, and both
    packages' errors summed over the sizes where both have a graph.

    Where a summary records the release card's verdicts (``card_verdict`` in
    :func:`comparisons`), the sums leave out the rows the card did not judge, and a method is
    left out against a package unless the card counts one of its rows there better and none
    worse, over every setting the homepage shows. So a comparison the card calls a tie, or worse
    anywhere, never reads as a win. Its ties stay in the sums, as every row does in the older
    summaries. A comparator in ``RESEARCH_CODE`` has no row here.
    """
    groups = time_groups(summaries)
    verdicts: dict[tuple[str, str], set] = {}
    for _, methods in groups:
        for m in methods:
            key = (m["spec"], m["comparator"])
            verdicts.setdefault(key, set()).update(r["card_verdict"] for r in m["rows"])

    def counted(key: tuple[str, str]) -> bool:
        seen = verdicts[key] - {None}
        return seen == {"unrecorded"} or ("better" in seen and "worse" not in seen)

    rows = []
    for name, methods in groups:
        by: dict[str, dict] = {}
        for c in error_comparisons(methods):
            if c["row"]["card_verdict"] is None or not counted(
                (c["method"]["spec"], c["method"]["comparator"])
            ):
                continue
            if c["method"]["comparator"] in RESEARCH_CODE:
                continue
            row = by.setdefault(
                c["method"]["comparator"],
                {"name": name, "spec": c["method"]["spec"], "family": c["method"]["family"],
                 "comparator": c["method"]["comparator"], "ours": 0.0, "theirs": 0.0},
            )  # fmt: skip
            row["ours"] += c["row"]["andrey_errors"]
            row["theirs"] += c["row"]["comparator_errors"]
        rows += by.values()
    return rows


def fewest_errors(summaries: list[dict]) -> float:
    """Andrey's largest cut in errors against a package compared, in percent, over
    :func:`accuracy_rows`: the README's accuracy line."""
    return max(-accuracy_change(r) for r in accuracy_rows(summaries))


def accuracy_change(row: dict) -> float:
    """Andrey's errors against the other package's in a row of ``accuracy_rows``, as a percent
    change: -50 is half as many."""
    return 100 * row["ours"] / row["theirs"] - 100 if row["theirs"] else 0.0


def accuracy_chart(summaries: list[dict]) -> str:
    """The accuracy cards on a belt."""
    return belt(accuracy_cards(summaries))


def notes_list(notes: list[str]) -> str:
    """The exceptions as a list under the chart; nothing when there are none."""
    if not notes:
        return ""
    return '<ul class="speed-notes">' + "".join(f"<li>{escape(n)}</li>" for n in notes) + "</ul>"


def speed_source(summary: dict) -> str:
    """Where the numbers come from, and what SHD measures."""
    return (
        f"Numbers from the benchmark summary generated on {escape(summary['generated'])}. SHD is "
        "the structural Hamming distance to the true graph; lower is better."
    )


def speed_table(summary: dict, *, notes: bool = True) -> str:
    """Every row of a summary, the chart's table-view equivalent, each comparison's errors in its
    own measure (:func:`error_kind`).

    With ``notes=False`` the exceptions and the source line are left out.
    """
    head = (
        "<tr><th scope='col'>d</th><th scope='col'>Datasets</th>"
        "<th scope='col' class='ct-ours'>Andrey</th>"
        "<th scope='col'>Other package</th><th scope='col'>Speedup</th>"
        "<th scope='col' class='ct-ours'>Andrey errors</th><th scope='col'>Other errors</th>"
        "<th scope='col'>Empty-graph errors</th></tr>"
    )
    body = []
    for method in ordered(summary):
        ours, theirs, empty = ERRORS[error_kind(method)][0]
        body.append(
            f"<tr><th scope='rowgroup' colspan='8'><strong>{escape(method['method'])}</strong> "
            f"with {escape(method['comparator'])}: {REGIMES[method['regime']]}; "
            f"{escape(method['hardware'])}; Andrey {escape(method['build'])}"
            "</th></tr>"
        )
        for r in method["rows"]:
            ratio = "-" if r["ratio"] is None else ratio_text(r["ratio"])
            cells = [
                f"{r['d']:,}",
                str(r["seeds"]),
                seconds_text(r["andrey_s"]),
                seconds_text(r["comparator_s"]),
                ratio,
                number_text(r[ours]),
                number_text(r[theirs]),
                "-" if empty is None else number_text(r[empty]),
            ]
            # Andrey's columns are tinted, and the lower time and the fewer errors are bold, as in
            # every comparison table on the site.
            classes = [[] for _ in cells]
            for k in (2, 5):
                classes[k].append("ct-ours")
            for k, (a, b) in ((2, (r["andrey_s"], r["comparator_s"])), (5, (r[ours], r[theirs]))):
                if a is not None and b is not None and a != b:
                    classes[k if a < b else k + 1].append("ct-best")
            body.append(
                "<tr>"
                + "".join(
                    f'<td class="{" ".join(c)}">{v}</td>' if c else f"<td>{v}</td>"
                    for v, c in zip(cells, classes)
                )
                + "</tr>"
            )
    listed_notes = notes_list(exceptions(summary)) if notes else ""
    source = f'<p class="generated">{speed_source(summary)}</p>' if notes else ""
    return (
        '<details class="speed-table"><summary>Show every measurement</summary>'
        f"{listed_notes}"
        f'<div class="table-scroll"><table><thead>{head}</thead><tbody>{"".join(body)}</tbody>'
        f"</table></div>{source}</details>"
    )


# ---- the method list -------------------------------------------------------------------------
def returns_text(spec) -> str:  # noqa: ANN001 -- an andrey.spec MethodSpec
    """What a method returns, in one or two words."""
    if spec.family == "pairwise":
        return "Two-node DAG"
    if spec.output.structure_type == "temporal":
        return "Temporal graph"
    if spec.output.graph_kind == "undirected":
        return "Undirected graph"
    return spec.output.graph_kind.upper()


# ---- the example, the wordmark, and the page --------------------------------------------------
def drawing(
    structure,
    label: str,
    positions: dict | None = None,
    size: tuple[int, int] | None = None,
) -> str:
    """Draw a structure with its column names, in the page's theme colors.

    ``size`` sets the drawing's own canvas; a smaller canvas shows the labels larger on the page.
    """
    width, height = size or (None, None)
    markup = str(
        andrey.viz.draw(
            structure,
            palette=TOKEN_PALETTE,
            background=False,
            positions=positions,
            width=width,
            height=height,
        )
    )
    ET.register_namespace("", "http://www.w3.org/2000/svg")
    svg = ET.fromstring(markup)
    svg.set("role", "img")
    svg.set("aria-label", label)
    return ET.tostring(svg, "unicode")


def run_script(name: str) -> tuple[str, str, dict]:
    """Run one of the site's example scripts; return its source, what it prints, and its names."""
    path = SITE / name
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        namespace = runpy.run_path(str(path), run_name="__main__")
    return path.read_text(encoding="utf-8"), printed.getvalue().strip(), namespace


def parts_layout(structure: GraphStructure) -> dict[int, tuple[float, float]]:
    """``andrey.viz.layout`` for each connected part of ``structure``, the parts in a grid, and a
    part laid out as a column turned into a row.

    ``layout`` layers nodes along directed edges, so a graph of small parts with few of those
    stacks into one tall column.
    """
    marks = structure.to_numpy()
    graph = nx.from_numpy_array((marks != 0).astype(int))
    parts = sorted(sorted(part) for part in nx.connected_components(graph))
    across = math.ceil(math.sqrt(len(parts)))
    gap = 0.5  # between parts, in a part's width
    positions = {}
    for i, nodes in enumerate(parts):
        row, column = divmod(i, across)
        part = GraphStructure.from_numpy(marks[np.ix_(nodes, nodes)], kind=structure.kind)
        local = andrey.viz.layout(part, engine="builtin")  # a ring for each undirected part
        if len({x for x, _ in local.values()}) == 1:
            local = {k: (y, x) for k, (x, y) in local.items()}
        for k, (x, y) in local.items():
            positions[nodes[k]] = (column * (1 + gap) + x, row * (1 + gap) + y)
    span = max(max(x for x, _ in positions.values()), max(y for _, y in positions.values())) or 1
    return {k: (x / span, y / span) for k, (x, y) in positions.items()}


def run_example() -> tuple[str, str, str]:
    """Run ``site/example.py``; return its source, what it prints, and its graph as an SVG."""
    code, printed, namespace = run_script("example.py")
    structure = namespace["out"].structure
    label = "The graph PC recovers from the Sachs data"
    return code, printed, drawing(structure, label, parts_layout(structure), size=(360, 260))


NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


# Each edge as [source, target, type], named by the CSV header: the rows oriented_edges() returns.
AGENT_QUERY = ".structure | .labels as $n | .edges[] | [$n[.source], $n[.target], .type]"


# What an agent runs, in order: find a method, read its parameters, run it. The jq filters are
# applied here, to the command line's own JSON; a test checks them against jq.
AGENT_LIST = "andrey list | jq -r '.[] | [.name, .output, .status] | @tsv'"
AGENT_HELP = (
    "andrey run pc --help --json | jq -c '.method.params.properties.alpha | {type, default}'"
)
AGENT_RUN = f"andrey run pc --data sachs.csv | jq -c '{AGENT_QUERY}'"
AGENT_LISTED = 3  # the list's first rows; the rest are counted


def run_cli(args: list[str]) -> dict | list:
    """Run the command line as a program would, so it answers in JSON; fail on a non-zero exit."""
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):  # not a terminal, so the answer is JSON
        status = andrey_cli(args)
    if status != 0:
        raise RuntimeError(f"andrey {' '.join(args)} exited with {status}")
    return json.loads(printed.getvalue())


def compact(value) -> str:  # noqa: ANN001
    """JSON as jq -c prints it."""
    return json.dumps(value, separators=(",", ":"))


def agent_frames() -> list[tuple[str, list[str], int]]:
    """Run the command line on the example's data as an agent would. Each frame is a command, the
    lines its jq filter prints, and how many of them are left out."""
    _, _, namespace = run_script("example.py")
    with tempfile.TemporaryDirectory(prefix="andrey-site-") as directory:
        data = Path(directory) / "sachs.csv"
        namespace["df"].to_csv(data, index=False)  # the header row names the nodes
        methods = run_cli(["list"])
        contract = run_cli(["run", "pc", "--help", "--json"])
        alpha = contract["method"]["params"]["properties"]["alpha"]
        structure = run_cli(["run", "pc", "--data", str(data)])["structure"]
    listed = ["\t".join((m["name"], m["output"], m["status"])) for m in methods]
    names = structure["labels"]
    edges = [[names[e["source"]], names[e["target"]], e["type"]] for e in structure["edges"]]
    return [
        (AGENT_LIST, listed[:AGENT_LISTED], len(listed) - AGENT_LISTED),
        (AGENT_HELP, [compact({"type": alpha["type"], "default": alpha["default"]})], 0),
        (AGENT_RUN, [compact(edge) for edge in edges], 0),
    ]


# The art each band of the homepage lies on (site/brand/art-<name>.jpg): the hero's, the band that
# leads from the families to the methods catalog, and the agents band's.
HERO_ART = "lanes"
METHODS_ART = "methods"
AGENTS_ART = "agentic"
PRIORITY = ' fetchpriority="high"'


def picture(path: str, attrs: str) -> str:
    """A brand JPEG (``_static/<path>.jpg``) with its AVIF copy first, which a browser that reads
    AVIF loads instead (``docs/conf.py`` writes the copies); ``attrs`` are the image's."""
    return (
        f'<picture><source type="image/avif" srcset="_static/{path}.avif">'
        f'<img src="_static/{path}.jpg" {attrs}></picture>'
    )


def band_art(art: str, *, priority: bool = False) -> str:
    """A band's art behind its type, with its description as the alt text; only the first
    screen's art loads before the rest."""
    load = PRIORITY if priority else ' loading="lazy"'
    return picture(
        f"art-{art}",
        f'class="band-art" alt="{escape(cover_alt(art), quote=True)}" width="1600" '
        f'height="640"{load}',
    )


# The methods catalog, which the hero, the families, and the next steps lead to.
CATALOG = "docs/methods.html"
# The release cards the homepage shows, newest first.
LATEST_CARDS = 4
RELEASE_CARDS = DOCS / "docs" / "benchmarks"


def release_versions() -> list[str]:
    """Every release card's version, newest first."""
    found = [p.stem for p in RELEASE_CARDS.glob("*.md") if re.fullmatch(r"\d+\.\d+\.\d+", p.stem)]
    return sorted(found, key=version_key, reverse=True)


def release_card(version: str) -> dict[str, str]:
    """A release card's version, theme (its title after the colon, if it has one), summary (its
    description after the dash), and cover: the release's own art in
    ``docs/_static/release-cards/``, or none, where the card shows the default cover."""
    text = (RELEASE_CARDS / f"{version}.md").read_text(encoding="utf-8")
    front = yaml.safe_load(text.split("---")[1])
    title = re.search(r"^# (.+)$", text, re.M)[1]
    theme = title.split(":", 1)[1].strip() if ":" in title else ""
    summary = " ".join(front["description"].split()).split(" - ", 1)[-1]
    art = DOCS / "_static" / "release-cards" / f"{version}.jpg"
    return {
        "version": version,
        "theme": theme[:1].upper() + theme[1:],
        "summary": summary[:1].upper() + summary[1:],
        "cover": f"_static/release-cards/{version}.jpg" if art.exists() else "",
    }


def latest_cards() -> str:
    """The newest ``LATEST_CARDS`` release cards as tiles: the cover, or the default cover's
    stipple ground with the version, then the version, the theme, and the summary."""
    out = []
    for card in map(release_card, release_versions()[:LATEST_CARDS]):
        version = card["version"]
        if card["cover"]:
            cover = (
                f'<img class="card-cover" src="{card["cover"]}" alt="" width="864" '
                'height="1119" loading="lazy">'
            )
        else:
            cover = f'<span class="card-cover card-ground" aria-hidden="true">{version}</span>'
        theme = f" <span>{escape(card['theme'])}</span>" if card["theme"] else ""
        out.append(
            f'<li><a href="{card_href(version)}">{cover}<span class="card-text"><b>{version}'
            f'{theme}</b><span class="card-summary">{escape(card["summary"])}</span></span>'
            "</a></li>"
        )
    return "".join(out)


# The prompt that installs the agent skill, as the README and the FAQ give it.
AGENT_PROMPT = (
    "Install Andrey's agent skill: make sure the andrey-core package is installed, run "
    "`andrey --skill`, and save its output as andrey/SKILL.md in your skills folder."
)


# Coding agents, and open models that drive them, as the pill shows them: each one's mono mark
# from Lobe Icons (site/agents/, MIT), drawn in the text's color. GLM's is Z.ai's.
AGENTS = {
    "claude": "Claude Code",
    "codex": "Codex",
    "cursor": "Cursor",
    "zai": "GLM",
    "deepseek": "DeepSeek",
}


def agent_logo(name: str) -> str:
    """``name``'s mark, its name as the tooltip."""
    svg = (SITE / "agents" / f"{name}.svg").read_text()
    paths = re.sub(r"<title>.*?</title>", "", re.search(r"<svg[^>]*>(.*)</svg>", svg, re.S)[1])
    return (
        '<svg class="agent-logo" viewBox="0 0 24 24" fill="currentColor" fill-rule="evenodd">'
        f"<title>{escape(AGENTS[name])}</title>{paths}</svg>"
    )


def agent_pill() -> str:
    """The pill in "Built for agents": it copies ``AGENT_PROMPT`` to paste into a coding agent
    (``prompt.js``), and without the script it leads to the FAQ answer that gives the prompt."""
    logos = "".join(agent_logo(name) for name in AGENTS)
    return (
        f'<a class="agent-pill" href="docs/faq.html#how-does-an-agent-use-it" '
        f'data-prompt="{escape(AGENT_PROMPT)}" title="Copy a prompt for your coding agent">'
        '<span class="agent-pill__label">Onboard your agent<span class="agent-pill__more"> to '
        'Andrey</span></span><span class="agent-logos" aria-hidden="true">'
        f"{logos}</span></a>"
    )


def round_down(value: float) -> int:
    """A ratio rounded down to one significant digit: 941 -> 900, 1,470 -> 1,000, 6.9 -> 6."""
    unit = 10 ** math.floor(math.log10(value))
    return int(value // unit * unit)


def more_than(value: float) -> tuple[str, float]:
    """A claim's number, the value rounded down to one significant digit (:func:`round_down`), or
    to a tenth below 10, with "More than" when the value clears it and "At least" when it equals
    it: 116 -> More than 100, 1.38 -> More than 1.3, 20 -> At least 20."""
    shown = round_down(value) if value >= 10 else math.floor(value * 10) / 10
    return ("More than" if value > shown else "At least"), shown


def terminal(frames: list[tuple[str, list[str], int]], shown: int = 3) -> str:
    """Commands and the first lines each prints, as in a terminal."""
    lines = []
    for command, printed, more in frames:
        lines.append(f'<span class="cmd">$ {escape(command)}</span>')
        lines += [escape(line) for line in printed[:shown]]
        if more or len(printed) > shown:
            lines.append(
                f'<span class="elide">... {more + max(len(printed) - shown, 0)} more</span>'
            )
    return f'<pre class="term">{chr(10).join(lines)}</pre>'


# ---- one featured method per family ---------------------------------------------------------
# Comparators that are a method's authors' research code, not a package: they stay in the charts
# and on their release cards, and they do not count among the packages a method is compared with.
RESEARCH_CODE = {
    "LOCI authors' code, LOCI-H",
    "SCORE authors' code",
    "NOTEARS authors' code",
    "authors' R RESIT",
}
# Families whose graphs beat the empty graph by link F1 (cross-variable links in the window graph,
# or adjacencies for LPCMCI), not by errors: a lag window can hold far more errors than the empty
# graph, whose F1 is 0.
F1_FAMILIES = {"temporal"}


def default_settings(summaries: list[dict]) -> dict[str, tuple[str, list[dict]]]:
    """Per registry method, the setting it is shown under and that setting's comparisons: the
    chart (:func:`time_groups`) whose name ``SHOWN_SETTINGS`` lists, else the shortest-named one,
    the first among equals. The name is the one ``SHOWN_SETTINGS`` gives, or the chart's."""
    found: dict[str, tuple[str, list[dict]]] = {}
    for name, methods in time_groups(summaries):
        shown = found.get(methods[0]["spec"])
        if shown is None or (
            shown[0] not in SHOWN_SETTINGS and (name in SHOWN_SETTINGS or len(name) < len(shown[0]))
        ):
            found[methods[0]["spec"]] = (name, methods)
    return {spec: (SHOWN_SETTINGS.get(n, n), ms) for spec, (n, ms) in found.items()}


def shared_rows(methods: list[dict]) -> dict[int, list[dict]]:
    """Each size at which every comparison in ``methods`` has both fit times and both graphs'
    errors, smallest first, with each comparison's row there: at the largest sample size where a
    lane measures several, the last listed among equal ones."""
    by_size = [{r["d"]: r for r in sorted(m["rows"], key=lambda r: r["n"] or 0)} for m in methods]
    keys = ("andrey_s", "comparator_s", "andrey_errors", "comparator_errors")
    out = {}
    for d in sorted(by_size[0]):
        rows = [sizes.get(d) for sizes in by_size]
        if all(row is not None and all(row[k] is not None for k in keys) for row in rows):
            out[d] = rows
    return out


def beats_empty(family: str, row: dict) -> bool:
    """Whether Andrey's graph beats the empty graph in ``row``: fewer errors than the empty graph,
    or in ``F1_FAMILIES`` a link F1 above 0."""
    if family in F1_FAMILIES:
        return (row["andrey_f1"] or 0) > 0
    return row["empty_errors"] is not None and row["andrey_errors"] < row["empty_errors"]


def compared_packages(summaries: list[dict]) -> dict[str, set[str]]:
    """Per supported registry method, the packages its published comparisons name, in any
    setting, by :func:`package_name`; ``RESEARCH_CODE`` is not a package."""
    out: dict[str, set[str]] = {}
    for m in supported(summaries):
        if m["comparator"] not in RESEARCH_CODE:
            out.setdefault(m["spec"], set()).add(package_name(m["comparator"]))
    return out


def feature_ranks(summaries: list[dict]) -> dict[str, tuple[int, int]]:
    """Each supported method that can be featured, by registry name: the packages it is compared
    with (:func:`compared_packages`) and its reach, the largest size at which every package in its
    shown setting (:func:`default_settings`) ran and Andrey's graph beats the empty graph
    (:func:`beats_empty`). A method with no such size is left out."""
    packages = compared_packages(summaries)
    ranks = {}
    for spec, (_, methods) in default_settings(summaries).items():
        family = methods[0]["family"]
        sizes = [d for d, rows in shared_rows(methods).items() if beats_empty(family, rows[0])]
        if sizes:
            ranks[spec] = (len(packages.get(spec, ())), sizes[-1])
    return ranks


def featured_among(ranks: dict[str, tuple[int, int]], names: list[str]) -> str | None:
    """The method to feature among ``names``: of those ``ranks`` holds (:func:`feature_ranks`),
    the one compared with the most packages, then the one with the largest reach, then the first
    in registry order; None when ``ranks`` holds none. The rule reads coverage, not the result of
    a comparison."""
    order = {s.name: i for i, s in enumerate(list_specs())}
    found = [name for name in names if name in ranks]
    return min(found, key=lambda n: (-ranks[n][0], -ranks[n][1], order[n]), default=None)


def featured_methods(summaries: list[dict]) -> dict[str, dict]:
    """The method each registry family features (:func:`featured_among` over the family's
    methods), in ``SPEC_FAMILIES`` order, as the homepage and the methods catalog show it; a family
    without one is left out. Each entry gives the method's registry name (``spec``), its shown
    ``name``, its ``packages``, and its reach as ``size`` in ``unit``, with the ``release`` and
    the ``(comparison, row)`` pairs at that size, Andrey's time and errors in each row."""
    ranks = feature_ranks(summaries)
    settings = default_settings(summaries)
    out = {}
    for family in SPEC_FAMILIES:
        spec = featured_among(ranks, [s.name for s in list_specs() if s.family == family])
        if spec is None:
            continue
        name, methods = settings[spec]
        packages, d = ranks[spec]
        out[family] = {
            "spec": spec,
            "name": name,
            "packages": packages,
            "size": d,
            "unit": methods[0]["unit"],
            "release": max((m["release"] for m in methods), key=version_key),
            "cells": list(zip(methods, shared_rows(methods)[d], strict=True)),
        }
    return out


def version_key(version: str) -> tuple[int, ...]:
    """A release's version for sorting: "0.10.0" after "0.9.0"."""
    return tuple(int(part) for part in version.split("."))


# How each row's note words the outcome at its size: the verdict its release card records
# (``card_verdict``: "tie" where the paired 95% interval of the difference in errors contains 0),
# or, where the card records none, the two error counts on the same datasets, as
# :func:`recorded_outcomes` reads them.
OUTCOME_WORDS = {
    "tie": ("Ties", ""),
    "better": ("Fewer errors", "than"),
    "worse": ("More errors", "than"),
    "same": ("Same errors", "as"),
    "fewer": ("Fewer errors", "than"),
    "more": ("More errors", "than"),
}


def outcome(row: dict) -> str:
    """The outcome a row records for Andrey against the other package: the release card's
    verdict, or else the error counts compared."""
    if row["card_verdict"] in ("tie", "better", "worse"):
        return row["card_verdict"]
    ours, theirs = row["andrey_errors"], row["comparator_errors"]
    return "same" if abs(ours - theirs) < 1e-9 else "fewer" if ours < theirs else "more"


def shape(kind: str, cx: float, cy: float, r: float) -> str:
    """A package's marker, as SVG path data centered on (cx, cy)."""
    if kind == "square":
        return f"M{cx - r:.1f} {cy - r:.1f}h{2 * r:.1f}v{2 * r:.1f}h{-2 * r:.1f}Z"
    if kind == "triangle":
        return f"M{cx:.1f} {cy - 1.2 * r:.1f}L{cx + r:.1f} {cy + 0.8 * r:.1f}H{cx - r:.1f}Z"
    if kind == "diamond":
        k = 1.3 * r
        return (
            f"M{cx:.1f} {cy - k:.1f}L{cx + k:.1f} {cy:.1f}"
            f"L{cx:.1f} {cy + k:.1f}L{cx - k:.1f} {cy:.1f}Z"
        )
    if kind == "cross":
        return (
            f"M{cx - r:.1f} {cy - r:.1f}L{cx + r:.1f} {cy + r:.1f}"
            f"M{cx + r:.1f} {cy - r:.1f}L{cx - r:.1f} {cy + r:.1f}"
        )
    if kind == "triangle_down":
        return f"M{cx:.1f} {cy + 1.2 * r:.1f}L{cx + r:.1f} {cy - 0.8 * r:.1f}H{cx - r:.1f}Z"
    tips = [math.pi / 2 + i * 4 * math.pi / 5 for i in range(5)]
    return (
        "M"
        + "L".join(
            f"{cx + 1.2 * r * math.cos(a):.1f} {cy - 1.2 * r * math.sin(a):.1f}" for a in tips
        )
        + "Z"
    )


# The marks of the other packages, in the order the rows first name them, so each package keeps
# one shape in every row; Andrey is the filled dot.
LEDGER_SHAPES = ("square", "diamond", "triangle", "cross", "star", "triangle_down")


def ledger_rows(featured: dict[str, dict]) -> list[dict]:
    """Each family's featured method (:func:`featured_methods`) as a ledger row: its name, family,
    size, release, Andrey's fit time, and each other package's time and outcome there
    (:func:`outcome`), the packages in the order their comparisons are listed."""
    rows = []
    for family, entry in featured.items():
        cells = entry["cells"]
        rows.append(
            {
                **entry,
                "family": family,
                "label": SPEC_FAMILIES[family],
                "seconds": cells[0][1]["andrey_s"],
                "others": [
                    (package_name(m["comparator"]), r["comparator_s"], outcome(r)) for m, r in cells
                ],
            }
        )
    return rows


def ledger_axis(rows: list[dict]) -> tuple[int, int]:
    """The rows' shared log axis of seconds: the whole decades around every time they show."""
    times = [t for row in rows for t in (row["seconds"], *(o[1] for o in row["others"]))]
    return math.floor(math.log10(min(times))), math.ceil(math.log10(max(times)))


def ledger_x(seconds: float, axis: tuple[int, int], width: float, pad: float) -> float:
    lo, hi = axis
    return pad + (math.log10(seconds) - lo) / (hi - lo) * (width - 2 * pad)


def ledger_mark(kind: str, x: float, y: float, ours: bool) -> str:
    """Andrey's filled dot, or another package's hollow shape (:func:`shape`)."""
    if ours:
        return f'<circle class="pt ours" cx="{x:.1f}" cy="{y:.1f}" r="5.5"/>'
    return f'<path class="pt theirs" d="{shape(kind, x, y, 4.6)}"/>'


def ledger_ticks(axis: tuple[int, int], width: int, pad: int) -> str:
    """The shared axis's labels, a decade apart, as an SVG strip of its own."""
    ticks = "".join(
        f'<text x="{ledger_x(10.0**e, axis, width, pad):.1f}" y="14" text-anchor="middle">'
        f"{10.0**e:,g} s</text>"
        for e in range(axis[0], axis[1] + 1)
    )
    return (
        f'<svg class="l-axis" viewBox="0 0 {width} 20" aria-hidden="true" focusable="false">'
        f"{ticks}</svg>"
    )


def ledger_strip(row: dict, marks: dict[str, str], axis: tuple[int, int], *, wide: bool) -> str:
    """One row's times on the shared axis: a mark per package and a band from Andrey's time to
    the slowest. The wide strip labels each mark with its package and time, in the first free
    lane above or below the line; the narrow one draws the marks only, its times in the text."""
    width, pad, height, y = (580, 30, 56, 28) if wide else (340, 24, 28, 14)
    points = [("Andrey", row["seconds"], True)] + [(p, t, False) for p, t, _ in row["others"]]
    xs = [ledger_x(t, axis, width, pad) for _, t, _ in points]
    said = ", ".join(f"{p} {seconds_text(t)}" for p, t, _ in points)
    label = f"Fit time of {row['name']} at {row['size']:,} {row['unit']}: {said}"
    out = [
        f'<svg class="l-strip" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{escape(label, quote=True)}">'
        if wide
        else f'<svg class="l-strip" viewBox="0 0 {width} {height}" aria-hidden="true" '
        'focusable="false">'
    ]
    for e in range(axis[0], axis[1] + 1):
        x = ledger_x(10.0**e, axis, width, pad)
        out.append(f'<line class="grid" x1="{x:.1f}" x2="{x:.1f}" y1="2" y2="{height - 2}"/>')
    out.append(f'<line class="band" x1="{xs[0]:.1f}" x2="{max(xs):.1f}" y1="{y}" y2="{y}"/>')
    for (package, _, ours), x in reversed(list(zip(points, xs, strict=True))):
        out.append(ledger_mark(marks.get(package, ""), x, y, ours))
    if wide:
        lanes: dict[int, list[tuple[float, float]]] = {y - 12: [], y + 20: []}
        for (package, seconds, ours), x in zip(points, xs, strict=True):
            text = f"{package} {seconds_text(seconds)}"
            w = len(text) * 12 * 0.56
            x0 = min(max(x - w / 2, 0), width - w)
            lane = next(
                (
                    ly
                    for ly, taken in lanes.items()
                    if all(x0 + w + 10 < a or x0 > b + 10 for a, b in taken)
                ),
                y + 20,
            )
            lanes[lane].append((x0, x0 + w))
            out.append(
                f'<text class="lbl{" ours" if ours else ""}" x="{x0:.1f}" y="{lane}">'
                f'{escape(package)} <tspan class="t">{seconds_text(seconds)}</tspan></text>'
            )
    return "".join(out) + "</svg>"


def ledger_note(featured: dict[str, dict]) -> str:
    """The ledger's footnote: how each row is chosen, and how its notes read, naming only the
    kinds of outcome the rows show."""
    kinds = {outcome(row) for entry in featured.values() for _, row in entry["cells"]}
    f1 = [entry["name"] for family, entry in featured.items() if family in F1_FAMILIES]
    text = (
        "Each family's method is the supported one compared with the most packages, at the "
        "largest size every package ran where Andrey's graph has fewer errors than the empty graph"
        + (f" ({listed(f1)}: a link F1 above 0)" if f1 else "")
        + "."
    )
    if kinds & {"tie", "better", "worse"}:
        text += (
            " A note is the verdict of the row's release card; it ties where the paired 95% "
            "interval of the difference in errors contains 0."
        )
    if kinds & {"same", "fewer", "more"}:
        text += (
            " Where the card records no verdict, the note compares the two packages' error "
            "counts on the same datasets."
        )
    return text


def ledger(featured: dict[str, dict]) -> str:
    """The featured methods, a row each on one log axis of seconds: the method, its family and
    size, and its release card; the times; and each other package's outcome there."""
    rows = ledger_rows(featured)
    axis = ledger_axis(rows)
    packages = dict.fromkeys(p for row in rows for p, _, _ in row["others"])
    if len(packages) > len(LEDGER_SHAPES):
        raise ValueError(f"give each of the {len(packages)} packages a shape in LEDGER_SHAPES")
    marks = dict(zip(packages, LEDGER_SHAPES, strict=False))

    def key(package: str) -> str:
        kind = "" if package == "Andrey" else marks[package]
        return (
            '<svg class="km" viewBox="0 0 14 14" aria-hidden="true" focusable="false">'
            f"{ledger_mark(kind, 7, 7, package == 'Andrey')}</svg>"
        )

    narrow = f'<div class="l-axis-sm" aria-hidden="true">{ledger_ticks(axis, 340, 24)}</div>'
    out = [
        '<div class="l-head" aria-hidden="true"><span>Method</span>'
        f"<div>{ledger_ticks(axis, 580, 30)}</div><span>Accuracy at that size</span></div>",
        narrow,
    ]
    for row in rows:
        notes = "".join(
            f'<li class="v v--{kind}"><b>{OUTCOME_WORDS[kind][0]}</b>'
            f"{' ' + OUTCOME_WORDS[kind][1] if OUTCOME_WORDS[kind][1] else ''} "
            f"{escape(package)}</li>"
            for package, _, kind in row["others"]
        )
        times = [("Andrey", row["seconds"])] + [(p, t) for p, t, _ in row["others"]]
        values = "".join(f"<li>{key(p)}{escape(p)} <b>{seconds_text(t)}</b></li>" for p, t in times)
        out.append(
            f'<div class="l-row" id="featured-{row["family"]}"><div class="l-name">'
            f'<div class="l-n1"><b>{escape(row["name"])}</b><a href="{card_href(row["release"])}">'
            f"{row['release']} card</a></div><span>{escape(row['label'])}, "
            f"{row['size']:,} {row['unit']}</span></div>"
            f'<div class="l-lg">{ledger_strip(row, marks, axis, wide=True)}</div>'
            f'<div class="l-sm">{ledger_strip(row, marks, axis, wide=False)}'
            f'<ul class="l-vals">{values}</ul></div><ul class="l-acc">{notes}</ul></div>'
        )
    return "".join(out) + narrow


def package_name(comparator: str) -> str:
    """The package ``comparator`` names, without a version or setting, from ``PACKAGE_NAMES``;
    a name that still holds a digit raises ``ValueError``."""
    name = PACKAGE_NAMES.get(comparator, comparator)
    if re.search(r"\d", name):
        raise ValueError(f"name the package of {comparator!r} in PACKAGE_NAMES")
    return name


def card_href(release: str) -> str:
    """The card for ``release``, or the release cards page if no card exists."""
    page = DOCS / "docs" / "benchmarks" / f"{release}.md"
    return f"docs/benchmarks/{release}.html" if page.exists() else "docs/benchmarks.html"


# The docs' code colors (pydata-sphinx-theme's defaults), so code reads the same on every page.
CODE_STYLES = {"light": "a11y-high-contrast-light", "dark": "a11y-high-contrast-dark"}


def code_html(source: str, language: str) -> str:
    """Source as highlighted HTML for a ``<pre class="hl"><code>``, by Pygments."""
    return highlight(source, get_lexer_by_name(language), HtmlFormatter(nowrap=True)).rstrip()


def code_css(scope: str, theme: str) -> str:
    """One theme's token colors under ``scope``; the page's own code-block frame stays as it is."""
    rules = HtmlFormatter(style=CODE_STYLES[theme]).get_style_defs(".hl").splitlines()
    return "\n".join(f"{scope}{rule}" for rule in rules if rule.startswith(".hl ."))


def tokens_css() -> str:
    """CSS custom properties for both palettes, and the code colors for each: the system theme, or
    the one the docs stored."""
    return "\n\n".join(
        [
            "/* Generated by site/build.py from andrey.viz.theme; do not edit. */",
            palette_block(":root", andrey.viz.LIGHT),
            code_css("", "light"),
            "@media (prefers-color-scheme: dark) {\n"
            + palette_block(':root:not([data-theme="light"])', andrey.viz.DARK)
            + "\n"
            + code_css(':root:not([data-theme="light"]) ', "dark")
            + "\n}",
            palette_block(':root[data-theme="dark"]', andrey.viz.DARK),
            code_css(':root[data-theme="dark"] ', "dark"),
        ]
    )


def homepage_code_css() -> str:
    """The code colors in both themes, keyed on the ``data-theme`` that the navbar's switch sets."""
    return code_css("", "light") + "\n" + code_css('html[data-theme="dark"] ', "dark")


# The methods the homepage's description names; every other registered method counts as "more".
DESCRIPTION_METHODS = ("PC", "GES", "FCI")


def page_description(summaries: list[dict]) -> str:
    """The homepage's summary for search results and link previews, as plain text."""
    registered = {spec.name for spec in list_specs()}
    if missing := [m for m in DESCRIPTION_METHODS if m.lower() not in registered]:
        raise ValueError(f"the description names unregistered methods: {', '.join(missing)}")
    named, more = ", ".join(DESCRIPTION_METHODS), len(registered) - len(DESCRIPTION_METHODS)
    return (
        f"Causal discovery for Python and AI agents. Learn causal graphs with {named} and "
        f"{more} more methods, on CPU or GPU. {CLAIM}"
    )


def abel_mark() -> str:
    """Abel's mark, the leaf of its logotype, as inline SVG in the text color around it."""
    logotype = (ROOT / "site" / "brand" / "abel-logo-white.svg").read_text(encoding="utf-8")
    paths = re.findall(r'<path d="([^"]+)"', logotype)
    if len(paths) != 4:  # the three letters, then the mark
        raise ValueError(f"expected the logotype's four paths, got {len(paths)}")
    return (
        '<svg class="abel-mark" viewBox="0 0 726 460" aria-hidden="true" focusable="false">'
        f'<path d="{paths[-1]}" fill="currentColor"/></svg>'
    )


def lede() -> str:
    """The hero's paragraph: the claim, then the supported methods and the families from the
    registry."""
    specs = list_specs()
    supported_count = sum(s.status == "supported" for s in specs)
    families = len({s.family for s in specs})
    return (
        f"<b>{escape(CLAIM)}</b> {supported_count} supported methods in "
        f"{NUMBER_WORDS[families]} families learn causal graphs from your data, on CPU or GPU, "
        "from Python or from an agent's command line."
    )


def cards_intro(summaries: list[dict]) -> str:
    """What the release cards hold: how many supported methods they compare with other
    packages, out of the registry's supported methods."""
    compared = len({m["spec"] for m in supported(summaries)})
    total = sum(s.status == "supported" for s in list_specs())
    share = f"all {total}" if compared == total else f"{compared} of the {total}"
    return (
        "Each row comes from a release card; together the cards compare "
        f"{share} supported methods with other packages."
    )


# Each family in one line, under its name in the families table (wide screens only).
FAMILY_LINES = {
    "constraint": "Tests which variables are independent, then removes and orients edges.",
    "score": "Searches for the graph that scores best on the data.",
    "permutation": "Searches over orderings of the variables and reads the graph from the best.",
    "functional": "Uses the shape of the noise or of the effect to direct every edge.",
    "temporal": "Finds lagged and same-step effects in time series and panels.",
    "latent": "Allows hidden common causes and shows where they may act.",
    "pairwise": "Decides which of two variables causes the other.",
}


def family_kinds() -> dict[str, dict[str, int]]:
    """For each family, in ``SPEC_FAMILIES`` order, how many of its supported methods read each
    kind of data (``MethodSpec.data_kinds``), in ``DATA_KINDS`` order."""
    specs = [s for s in list_specs() if s.status == "supported"]
    return {
        family: {
            k: sum(s.family == family and k in s.data_kinds for s in specs) for k in DATA_KINDS
        }
        for family in SPEC_FAMILIES
    }


def families_table() -> str:
    """The families, by the data they read: a row per family, linked to its table on the methods
    page, and a column per kind of data, linked to its row there; a dot in each cell, sized by the
    number of supported methods it counts."""
    specs = [s for s in list_specs() if s.status == "supported"]
    head = "".join(
        f'<th scope="col"><a href="{CATALOG}#{kind_anchor(k)}"><span>'
        f"{escape(DATA_KIND_TEXT[k][0])}</span></a></th>"
        for k in DATA_KINDS
    )
    rows = []
    for family, counts in family_kinds().items():
        total = sum(s.family == family for s in specs)
        cells = "".join(
            f'<td><span class="dot" style="--n:{n}">{n}</span></td>'
            if n
            else '<td class="none"><span class="visually-hidden">0</span></td>'
            for n in counts.values()
        )
        rows.append(
            f'<tr><th scope="row"><a href="{CATALOG}#family-{family}">'
            f"{escape(SPEC_FAMILIES[family])}</a>"
            f'<span class="ct">{total}<span class="visually-hidden"> supported methods</span>'
            f'</span><span class="ln">{escape(FAMILY_LINES[family])}</span></th>{cells}</tr>'
        )
    return (
        '<table class="matrix"><thead><tr><th scope="col"><span class="visually-hidden">Family'
        f"</span></th>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"
    )


def homepage_context(summary: dict, *, placeholder: bool = False) -> dict[str, str]:
    """Everything the homepage template shows, as HTML, from the first release's summary and the
    later releases' published ones."""
    fastest(summary)
    summaries = release_summaries(summary)
    featured = featured_methods(summaries)
    code, printed, figure = run_example()
    banner = (
        '<p class="placeholder-banner" role="note">Placeholder numbers, copied from the '
        "documentation's benchmarks page. Not for release.</p>"
    )
    return {
        "title": escape(SEARCH_TITLE),
        # Plain text: docs/conf.py escapes it into the page's description and link-preview tags.
        "description": page_description(summaries),
        "robots": '<meta name="robots" content="noindex">' if placeholder else "",
        "banner": banner if placeholder else "",
        "lede": lede(),
        # The first screen's art, which the page asks for before its stylesheets load.
        "hero_image": f"_static/art-{HERO_ART}.avif",
        "hero_art": band_art(HERO_ART, priority=True),
        "methods_art": band_art(METHODS_ART),
        "agents_art": band_art(AGENTS_ART),
        "ledger": ledger(featured),
        "ledger_note": escape(ledger_note(featured)),
        "catalog": CATALOG,
        "families": families_table(),
        "cards_intro": cards_intro(summaries),
        "cards": latest_cards(),
        "card_count": str(len(release_versions())),
        "example_code": code_html(code.rstrip(), "python"),
        "example_output": escape(printed),
        "example_figure": figure,
        "agent_pill": agent_pill(),
        "terminal": terminal(agent_frames()),
        "style": homepage_code_css(),
    }


# ---- the method to start with on each kind of data ---------------------------------------------
# The kinds of data a method that reads them cannot do without: one its result implies, or a list
# of datasets. A required ``env`` input is the other way a method needs a kind (:func:`needs`).
NEEDED_KINDS = {"datasets", "hidden_causes", "time_series", "two_variables"}


def needs(spec, kind: str) -> bool:  # noqa: ANN001 -- an andrey.spec MethodSpec
    """Whether ``spec`` cannot run without data of ``kind``: a required ``env`` input, or a kind
    in ``NEEDED_KINDS`` that it reads."""
    if kind == "environments":
        return any(i.name == "env" and i.required for i in spec.data.inputs)
    return kind in NEEDED_KINDS and kind in spec.data_kinds


def start_with(ranks: dict[str, tuple[int, int]], kind: str) -> str | None:
    """The method to start with on data of ``kind``, from :func:`feature_ranks`: in the first
    family, in ``SPEC_FAMILIES`` order, where :func:`featured_among` finds one, its pick among the
    methods that need the kind (:func:`needs`), or among those that read it when none needs it."""
    for family in SPEC_FAMILIES:
        readers = [s for s in list_specs() if s.family == family and kind in s.data_kinds]
        needing = [s.name for s in readers if needs(s, kind)]
        pick = featured_among(ranks, needing or [s.name for s in readers])
        if pick:
            return pick
    return None


# ---- the methods catalog (docs/methods.html; its links are relative to docs/) --------------------
# Each kind of data (andrey.spec.DATA_KINDS): its label and the line that says what it means.
DATA_KIND_TEXT = {
    "continuous": ("Continuous", "Numbers on a scale, one column per variable"),
    "categorical": ("Categorical", "Labels, such as yes and no, coded as whole numbers"),
    "mixed": ("Mixed", "Some columns numbers, some labels"),
    "missing": ("Missing values", "Some cells are empty"),
    "environments": (
        "Several environments",
        "Rows from experiments, interventions, or sites, each labeled",
    ),
    "datasets": ("Several datasets", "One table per group or time point, same columns"),
    "hidden_causes": ("Hidden causes", "Some common causes are not measured"),
    "time_series": ("Time series", "One row per time step"),
    "two_variables": ("Two variables", "Which one causes the other?"),
}
# Each graph a method returns (returns_text): its section of the graph types guide.
GRAPH_SECTIONS = {
    "DAG": "dag-a-directed-model",
    "CPDAG": "cpdag-an-equivalence-class",
    "PAG": "pag-uncertainty-with-hidden-variables",
    "Undirected graph": "undirected-graphs-adjacency-without-direction",
    "Temporal graph": "temporal-graphs",
    "Two-node DAG": "dag-a-directed-model",
}


def first_clause(summary: str) -> str:
    """The registry summary up to its first colon, semicolon, or sentence end."""
    return re.split(r"(?<=[a-z)]{2})\. |[:;] ", summary, maxsplit=1)[0].rstrip(".")


def purpose(spec) -> str:  # noqa: ANN001 -- an andrey.spec MethodSpec
    """What a method is for, in one line: the registry summary's first clause, or what follows
    its colon where that clause is only the method's name ("NOTEARS: least squares ...")."""
    text = first_clause(spec.summary)
    if len(text.split()) <= 2 and ":" in spec.summary:
        text = re.split(r"[;.] |, by ", spec.summary.split(":", 1)[1].strip())[0]
    return text[0].upper() + text[1:]


def kind_anchor(kind: str) -> str:
    return "data-" + kind.replace("_", "-")


def compared(summaries: list[dict]) -> dict[str, tuple[list[str], list[str]]]:
    """Each supported method's published comparisons: what it is compared with, packages and
    authors' code by :func:`package_name` in the summaries' order, and the releases whose cards
    hold them, oldest first."""
    packages: dict[str, dict[str, None]] = {}
    releases: dict[str, set[str]] = {}
    for method in supported(summaries):
        packages.setdefault(method["spec"], {})[package_name(method["comparator"])] = None
        releases.setdefault(method["spec"], set()).add(method["release"])
    return {
        name: (list(packages[name]), sorted(releases[name], key=version_key)) for name in packages
    }


def card_link(release: str) -> str:
    """A link to ``release``'s card from docs/, or to the release cards page if it has none."""
    return card_href(release).removeprefix("docs/")


def cards_text(releases: list[str]) -> str:
    links = [f'<a href="{card_link(r)}">{r}</a>' for r in releases]
    return f"{listed(links)} release card{'s' if len(links) > 1 else ''}"


def display_name(name: str, names: dict[str, str]) -> str:
    """A method's name as a table shows it: its prose name, capitalized ("Hill climbing")."""
    prose = names[name]
    return prose[0].upper() + prose[1:]


def method_cell(spec, names: dict[str, str]) -> str:  # noqa: ANN001 -- an andrey.spec MethodSpec
    """A method's name, linked to its API entry, and its function in code."""
    call = spec.name.replace("_", "_<wbr>")
    return (
        f'<a href="api/andrey.{spec.name}.html">{escape(display_name(spec.name, names))}</a> '
        f'<code class="docutils literal notranslate">andrey.{call}</code>'
    )


def table_html(head: list[str], rows: list[str]) -> str:
    """A table in the documentation's style, which stacks each row on a phone."""
    cells = "".join(f'<th scope="col">{label}</th>' for label in head)
    return (
        '<div class="pst-scrollable-table-container"><table class="table stacked">'
        f"<thead><tr>{cells}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


def catalog_start(summaries: list[dict], names: dict[str, str]) -> str:
    """The catalog's start table: one row per kind of data a method reads (``data-<kind>``), with
    what the kind means, the method to start with (:func:`start_with`), the graph it returns, and
    how many methods of each family read the kind, one family a line."""
    specs = list_specs()
    ranks = feature_ranks(summaries)
    rows = []
    for kind in DATA_KINDS:
        readers = [s for s in specs if kind in s.data_kinds]
        if not readers:
            continue
        label, hint = DATA_KIND_TEXT[kind]
        start = get_spec(start_with(ranks, kind))
        graph = returns_text(start)
        families = [
            f'<span class="nowrap"><a href="#family-{family}">{family_label}</a> '
            f"{sum(s.family == family for s in readers)}</span>"
            for family, family_label in SPEC_FAMILIES.items()
            if any(s.family == family for s in readers)
        ]
        rows.append(
            f'<tr id="{kind_anchor(kind)}"><td><strong>{label}</strong><br>{hint}</td>'
            f"<td>{method_cell(start, names)}</td>"
            f'<td><a href="guides/graph-types.html#{GRAPH_SECTIONS[graph]}">{graph}</a></td>'
            f"<td>{'<br>'.join(families)}</td></tr>"
        )
    return table_html(["Your data", "Start with", "Returns", "Methods that read it"], rows) + "\n"


def catalog_families(summaries: list[dict], names: dict[str, str]) -> str:
    """The catalog's family tables, as MyST: one section per family (``family-<family>``),
    opening with the method it features (:func:`featured_methods`), and one row per method
    (``method-<name>``): its name, an Experimental badge where it is experimental, and what it
    is for, then the kinds of data it reads, the graph it returns, and its comparisons with other
    packages."""
    specs = list_specs()
    starts = {family: entry["spec"] for family, entry in featured_methods(summaries).items()}
    found = compared(summaries)
    badge = ' <span class="sd-badge sd-outline-muted sd-text-muted">Experimental</span>'
    out = []
    for family, label in SPEC_FAMILIES.items():
        members = [s for s in specs if s.family == family]
        if not members:
            continue
        rows = []
        for s in members:
            labels = [DATA_KIND_TEXT[k][0] for k in s.data_kinds]
            reads = ", ".join([labels[0], *(text.lower() for text in labels[1:])])
            if s.name in found:
                packages, releases = found[s.name]
                comp = f"{escape(listed(packages))}<br>{cards_text(releases)}"
            else:
                comp = "Not compared yet"
            rows.append(
                f'<tr id="method-{s.name}"><td>{method_cell(s, names)}'
                f"{badge if s.status == 'experimental' else ''}<br>{escape(purpose(s))}</td>"
                f"<td>{reads}</td><td>{returns_text(s)}</td><td>{comp}</td></tr>"
            )
        experimental = sum(s.status == "experimental" for s in members)
        count = f"{len(members)} method{'s' if len(members) > 1 else ''}"
        count += f", {experimental} experimental" if experimental else ""
        start = (
            f' Start with <a href="#method-{starts[family]}">'
            f"{escape(display_name(starts[family], names))}</a>."
            if family in starts
            else ""
        )
        out += [
            f"(family-{family})=",
            f"### {label}",
            "",
            "```{raw} html",
            f"<p>{count}.{start}</p>"
            + table_html(["Method", "Reads", "Returns", "Compared with"], rows),
            "```",
            "",
        ]
    return "\n".join(out)


# ---- compared with other packages (docs/comparison.html; its links are relative to docs/) -------
# Andrey beside other packages. Feature marks come from released code
# (causal-learn 0.1.4.8, Tetrad and causal-cmd 7.6.10, pcalg 2.7-12, bnlearn 5.2.1). Hooks for
# custom tests or scores do not count as implementations. Features only on a development branch
# do not count as released. Permutation invariance uses the 0.7.0 post's measurements; open
# benchmarks use each package's README, documentation, and release notes. A real-data feature is
# a yes when the package's main learners have it, and partly when only some do. Interventional
# data is a yes when a method takes each row's environment or the intervention targets.
PEERS = ("Andrey", "causal-learn", "Tetrad", "pcalg", "bnlearn")
PEER_VERSIONS = "causal-learn 0.1.4.8, Tetrad 7.6.10, pcalg 2.7-12, and bnlearn 5.2.1"
# Each mark's word in the table.
MARKS = {"yes": "Yes", "part": "Partly", "no": "No"}
# Table sections in order, each with measured rows (``measured_rows``) followed by feature marks.
SECTIONS = ("Coverage", "Real data", "Speed", "Agentic")
# Each entry holds a section, an attribute, marks in PEERS order, and notes keyed by package
# for every "part" mark. Attribute labels are brief; the notes give details.
COMPARISON = (
    ("Coverage", "Temporal data", ("yes", "yes", "yes", "no", "no"), {}),
    (
        "Coverage",
        "Mixed data",
        ("yes", "part", "yes", "no", "yes"),
        {
            "causal-learn": "No mixed-data test or score; only KCI and the generalized score "
            "take category codes, and neither is built for them.",
        },
    ),
    ("Coverage", "Nonlinear data", ("yes", "yes", "yes", "no", "no"), {}),
    (
        "Coverage",
        "Interventional data",
        ("yes", "part", "part", "yes", "part"),
        {
            "causal-learn": "CD-NOD only, for data from several domains, with no targets.",
            "Tetrad": "Intervention variables tagged in a metadata file join the first knowledge "
            "tier as ordinary variables; no method takes the targets.",
            "bnlearn": "The mbde score only, for categorical data in its score-based searches.",
        },
    ),
    (
        "Real data",
        "Missing values",
        ("yes", "part", "yes", "no", "yes"),
        {"causal-learn": "PC and CD-NOD only, with the Fisher-z test."},
    ),
    (
        "Real data",
        "Background knowledge",
        ("yes", "part", "yes", "part", "yes"),
        {
            "causal-learn": "PC, FCI, CD-NOD, the LiNGAM methods, and exact search; GES, BOSS, "
            "and GRaSP take none.",
            "pcalg": "Fixed adjacencies only, with no directions or tiers.",
        },
    ),
    (
        "Real data",
        "Edge confidence",
        ("yes", "part", "yes", "no", "yes"),
        {"causal-learn": "Bootstrap edge probabilities for the LiNGAM methods only."},
    ),
    (
        "Speed",
        "CPU parallelization",
        ("yes", "part", "part", "part", "part"),
        {
            "causal-learn": "Several CPU cores for RLCD and the FastKCI test only.",
            "Tetrad": "Several threads for FGES, BOSS, GFCI, and PC's adjacency search.",
            "pcalg": "Several cores for the skeleton step of its stable.fast method only.",
            "bnlearn": "A cluster for its constraint-based methods and DirectLiNGAM only.",
        },
    ),
    (
        "Speed",
        "GPU acceleration",
        ("yes", "part", "no", "no", "no"),
        {"causal-learn": "CALM only."},
    ),
    (
        "Coverage",
        "Permutation invariance",
        ("yes", "no", "part", "no", "no"),
        {
            "Tetrad": "PC and FCI follow the variables' names; BOSS, GRaSP, and GIN changed with "
            "the column order.",
        },
    ),
    (
        "Agentic",
        "JSON command line",
        ("yes", "no", "part", "no", "no"),
        {"Tetrad": "causal-cmd writes the graph, and only the graph, to a JSON file."},
    ),
    (
        "Agentic",
        "Schema for every method",
        ("yes", "no", "part", "no", "no"),
        {"Tetrad": "Typed parameter defaults and bounds, in an HTML manual."},
    ),
    (
        "Agentic",
        "Agent skill",
        ("yes", "no", "part", "no", "no"),
        {"Tetrad": "py-tetrad's development branch has a skill and guides; no release has them."},
    ),
    (
        "Speed",
        "Open benchmarks",
        ("yes", "no", "no", "no", "no"),
        {},
    ),
)
# Distinct methods per package, each a published algorithm that returns a whole graph, counted
# once: a variant chosen by an argument, a multi-dataset or instance-specific version, another
# score or test in the same search, a wrapper, and a search that only orients a given graph count
# with the method they extend, or not at all. Andrey's count is the supported methods of its
# registry: gies runs ges's search, multi_group_direct_lingam extends direct_lingam, inter_iamb is a
# variant of iamb, and mmhc is rsmax2 with its default phases, as bnlearn's are counted. The others
# are read from each package's released code.
COUNTED_WITH = {
    "gies": "ges",
    "multi_group_direct_lingam": "direct_lingam",
    "inter_iamb": "iamb",
    "mmhc": "rsmax2",
}
METHOD_COUNTS = {
    "Andrey": sum(s.status != "experimental" and s.name not in COUNTED_WITH for s in list_specs()),
    "causal-learn": 21,
    "Tetrad": 21,
    "pcalg": 8,
    "bnlearn": 10,
}


def measured_rows() -> list[tuple[str, str, list[str]]]:
    """Measured rows as (section, label, values in PEERS order). Speed has no pooled value across
    methods; the release cards compare each method separately."""
    return [("Coverage", "Distinct methods", [str(METHOD_COUNTS[p]) for p in PEERS])]


def comparison_notes() -> list[tuple[str, str, str, str]]:
    """Check the table and number its notes: (feature, package, mark, note) in reading order."""
    notes = []
    for _, feature, marks, row_notes in COMPARISON:
        if len(marks) != len(PEERS) or set(marks) - MARKS.keys():
            raise ValueError(f"{feature!r}: give one of {sorted(MARKS)} for each of {PEERS}")
        if set(row_notes) - set(PEERS):
            unknown = set(row_notes) - set(PEERS)
            raise ValueError(f"{feature!r}: notes name unknown packages {unknown}")
        for package, mark in zip(PEERS, marks):
            if mark == "part" and package not in row_notes:
                raise ValueError(f"{feature!r}: say what {package} covers in part")
            if package in row_notes:
                notes.append((feature, package, mark, row_notes[package]))
    return notes


def comparison_source() -> str:
    """What the table's marks were read from."""
    return f"Read from each package's released code: {PEER_VERSIONS}."


def comparison_page() -> str:
    """The comparison page's table and notes, as MyST: a column per package in ``PEERS``, Andrey's
    first and tinted, a group of rows per section, each mark a word, each partial mark numbered to
    its note under the table, then how the distinct methods are counted."""
    # The notes in reading order: by section, then as COMPARISON lists them.
    section = {feature: sec for sec, feature, _, _ in COMPARISON}
    notes = sorted(comparison_notes(), key=lambda n: SECTIONS.index(section[n[0]]))
    numbers = {(f, p): k for k, (f, p, _, _) in enumerate(notes, 1)}
    ours = ' class="ct-ours"'
    head = "".join(
        f'<th scope="col"{ours if p == "Andrey" else ""}>{escape(p)}</th>' for p in PEERS
    )
    measured = measured_rows()
    # Every body cell is a td, so the rows keep the documentation table's card; a section's name
    # heads its rows in bold.
    rows = []
    for group in SECTIONS:
        rows.append(f'<tr><td colspan="{1 + len(PEERS)}"><strong>{group}</strong></td></tr>')
        for _, label, values in (r for r in measured if r[0] == group):
            cells = "".join(
                f"<td{ours if p == 'Andrey' else ''}>{v}</td>"
                for p, v in zip(PEERS, values, strict=True)
            )
            rows.append(f"<tr><td>{escape(label)}</td>{cells}</tr>")
        for _, feature, marks, _ in (r for r in COMPARISON if r[0] == group):
            cells = []
            for package, mark in zip(PEERS, marks, strict=True):
                k = numbers.get((feature, package))
                ref = f'<sup><a href="#comparison-note-{k}">{k}</a></sup>' if k else ""
                cells.append(f"<td{ours if package == 'Andrey' else ''}>{MARKS[mark]}{ref}</td>")
            rows.append(f"<tr><td>{escape(feature)}</td>{''.join(cells)}</tr>")
    listed_notes = "".join(
        f'<li id="comparison-note-{k}"><strong>{escape(feature)}, {escape(package)}.</strong> '
        f"{escape(note)}</li>"
        for k, (feature, package, _, note) in enumerate(notes, 1)
    )
    supported_count = sum(s.status == "supported" for s in list_specs())
    pairs = listed([f"`{name}` with `{base}`" for name, base in COUNTED_WITH.items()])
    return "\n".join(
        [
            "```{raw} html",
            '<div class="pst-scrollable-table-container"><table class="table">'
            f"<thead><tr><td></td>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>",
            "```",
            "",
            comparison_source(),
            "",
            "## Notes",
            "",
            "```{raw} html",
            f"<ol>{listed_notes}</ol>",
            "```",
            "",
            "## How the table is read",
            "",
            "- **Distinct methods** counts each published algorithm that returns a whole graph "
            "once. A variant chosen by an argument, a version for several datasets, another score "
            "or test in the same search, and a wrapper count with the method they extend; a "
            f"search that only orients a given graph does not count. Andrey's {supported_count} "
            f"supported methods count as {METHOD_COUNTS['Andrey']}: {pairs}. Experimental methods "
            "do not count.",
            "- A feature counts when a released version has it. A hook for your own test or score "
            "is not an implementation, and a feature only on a development branch is not released.",
            "- A real-data feature is a yes when the package's main learners have it, and partly "
            "when only some do. Interventional data is a yes when a method takes each row's "
            "environment or the intervention targets.",
            "- Permutation invariance comes from the measurements in the "
            "[0.7.0 post](../blog/20261019-andrey-0-7-0.md#how-andrey-compares), and open "
            "benchmarks from each package's README, documentation, and release notes.",
            "",
        ]
    )

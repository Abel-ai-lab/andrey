"""The homepage's content, generated from the release summaries and the method registry.

``docs/conf.py`` fills the homepage template (``site/homepage.html``) with
:func:`homepage_context`. ``docs/DEVELOPMENT.md`` has the build command.

Every number on the homepage comes from a release's published benchmark summary (``RELEASES``) or
from the method registry. A method measured against the same package in two summaries shows the
newer summary's rows (:func:`newest`). The speed and accuracy sections show one method per
registry family, the one :func:`featured_methods` picks, beside every package compared with it. The
README, the live demo, and the homepage's description state ``CLAIM``, which carries no number;
:func:`claim_sentence` words the first summary's headline. The method counts come from the method
registry. A summary outside ``benchmarks/published/`` loads only as a placeholder, which marks
the page as not for release. Links are relative to the site root.
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
import warnings
import xml.etree.ElementTree as ET
from collections import Counter
from html import escape
from pathlib import Path
from typing import get_args

import networkx as nx
import numpy as np
from pygments import highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import get_lexer_by_name

import andrey.viz
from andrey import GraphStructure
from andrey.cli import main as andrey_cli
from andrey.core.structure import Kind
from andrey.spec import DATA_KINDS, get_spec, list_specs
from andrey.spec.concepts import BLANKET_STRUCTURES

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
_why_fast = runpy.run_path(str(SITE / "why_fast.py"))
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
# leaves them to the card too. PCMB and CMB have no reference implementation: their rows sit
# beside their papers' published results, CMB's also beside global PC read at the target, with no
# comparator, on their release card.
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
    "pcmb.paper",
    "cmb.paper",
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
    if spec.output.graph_kind == "digraph":
        return "Directed graph"
    return spec.output.graph_kind.upper()


def first_clause(summary: str) -> str:
    """The registry summary up to its first colon, semicolon, or sentence end."""
    return re.split(r"(?<=[a-z)]{2})\. |[:;] ", summary, maxsplit=1)[0].rstrip(".")


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


# The three graphs site/interface.py draws from one table, in the order the page shows them: each
# method's name, the kind of graph it returns, and what that kind says about the sprinkler network.
# The methods answer differently only on season-rain and season-sprinkler, and the drawings' edge
# colors (undirected, circle, directed) carry the difference.
INTERFACE_CAPTIONS = {
    "direct_lingam": (
        "DirectLiNGAM",
        "DAG",
        "One direction for every edge: non-Gaussian noise settles each one.",
    ),
    "pc": (
        "PC",
        "CPDAG",
        "The data cannot say which way the two edges from season point, so they stay plain lines.",
    ),
    "fci": (
        "FCI",
        "PAG",
        "Circles on both ends: either direction, or a hidden common cause, fits the data.",
    ),
}
# The two drawings of VARMA-LiNGAM's graph over the lawn's days, after the three.
TEMPORAL_CAPTIONS = (
    (
        "Graph over time",
        "from VARMA-LiNGAM. Rain wets the lawn the same day; the sprinkler wets it the next day, "
        "and a dry lawn turns the sprinkler on the next day.",
    ),
    (
        "Cyclic graph",
        "the same graph with time collapsed. The sprinkler and the lawn drive each other, though "
        "every arrow over time points forward.",
    ),
)
# Every kind a supported method returns as a GraphStructure (andrey.core.structure.Kind), in the
# homepage's words, then the graphs over time a TemporalStructure holds. A MAG comes from a
# conversion, GraphStructure.to_mag, never from a method. The homepage shows supported methods only,
# so a kind that only experimental methods return is not on it. A skeleton, the adjacencies of a
# causal graph, is not itself a causal graph, so the homepage does not name it: MMPC returns one by
# default, and the blanket methods with structure="skeleton" or "moral".
GRAPH_KINDS = {"dag": "DAGs", "cpdag": "CPDAGs", "pag": "PAGs", "digraph": "cyclic graphs"}
CONVERSION_KINDS = {"mag"}
SKELETON_KINDS = {"undirected"}
TEMPORAL_KIND = "graphs over time"
NUMBER_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def experimental_kinds() -> set[str]:
    """The graph kinds that only experimental methods return: the kinds their specs declare and
    the kinds of the graphs their ``structure`` can name."""
    specs = list_specs()
    supported = {spec.output.graph_kind for spec in specs if spec.status == "supported"}
    if any(s.status == "supported" and s.output.structure_type == "temporal" for s in specs):
        supported.add("digraph")  # a TemporalStructure's summary graph
    experimental = {spec.output.graph_kind for spec in specs if spec.status == "experimental"}
    for spec in specs:
        structures = [p.choices for p in spec.params if p.concept == "structure" and p.choices]
        if spec.status == "experimental" and structures:
            experimental |= {BLANKET_STRUCTURES[choice] for choice in structures[0]}
    return experimental - supported


def graph_kinds() -> list[str]:
    """Every kind of causal graph the supported methods return, in the homepage's words."""
    skipped = CONVERSION_KINDS | SKELETON_KINDS | experimental_kinds()
    kinds = [kind for kind in get_args(Kind) if kind not in skipped]
    if set(kinds) != GRAPH_KINDS.keys():
        raise ValueError(f"name every graph kind in GRAPH_KINDS: {sorted(kinds)}")
    return [GRAPH_KINDS[kind] for kind in kinds] + [TEMPORAL_KIND]


def graph_title() -> str:
    """The graphs tile's and dialog's title: how many kinds of causal graph, in words."""
    return f"{NUMBER_WORDS[len(graph_kinds())].capitalize()} kinds of causal graph"


# Where each of the lawn's variables sits within its day, as (column, row): the rain left of the
# lawn it wets, the sprinkler above the lawn, so the edges between the days cross clear of the
# labels under the nodes.
DAY_SPOTS = {"rain": (0, 0.45), "wet": (1, 0.45), "sprinkler": (1, 0)}


def two_days(temporal) -> tuple[GraphStructure, dict[int, tuple[float, float]]]:
    """A graph over time with lags 0 and 1 drawn as two days side by side, each variable once a
    day (``DAY_SPOTS``): the same-day edges within each day and the lag-1 edges from the first day
    into the second."""
    if max(temporal.lags) > 1:
        raise ValueError(f"two days hold lags 0 and 1, not {temporal.lags}")
    summary = temporal.summary_graph()
    p = len(temporal.labels)
    marks = np.zeros((2 * p, 2 * p), dtype=int)
    for i in range(p):
        for j in range(p):
            for lag in summary.lags_of(i, j):
                arrows = [(i, j), (p + i, p + j)] if lag == 0 else [(i, p + j)]
                for a, b in arrows:
                    marks[a, b], marks[b, a] = andrey.viz.TAIL, andrey.viz.ARROW
    labels = [f"{name}(t-1)" for name in temporal.labels]
    labels += [f"{name}(t)" for name in temporal.labels]
    columns = 1 + max(x for x, _ in DAY_SPOTS.values())  # a day's
    positions = {
        day * p + i: ((day * columns + x) / (2 * columns - 1), y)
        for day in (0, 1)
        for i, (x, y) in enumerate(DAY_SPOTS[name] for name in temporal.labels)
    }
    return GraphStructure.from_numpy(marks, kind="dag", labels=labels), positions


def run_interface() -> tuple[str, str, str]:
    """Run ``site/interface.py``; return the loop it shows, what it prints, and a figure for each
    kind of causal graph: one per method on the lawn's table, then the graph over the lawn's days
    and the same graph with time collapsed."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)  # VARMA-LiNGAM is experimental
        source, printed, namespace = run_script("interface.py")
    graphs = namespace["graphs"]
    if graphs.keys() != INTERFACE_CAPTIONS.keys():
        raise ValueError(f"caption every method site/interface.py runs: {sorted(graphs)}")
    # The page shows the loop; the line that keeps each graph for the drawings is the build's.
    loop = source[source.index("for learn in") : source.index("\n\n", source.index("for learn in"))]
    shown = "\n".join(line for line in loop.splitlines() if "graphs[" not in line)
    # One layout for all three, from the DAG, so the drawings differ only in their edge marks.
    positions = andrey.viz.layout(graphs["direct_lingam"])
    for key, (name, kind, _) in INTERFACE_CAPTIONS.items():
        if graphs[key].kind != kind.lower():
            raise ValueError(f"{name} returns a {graphs[key].kind}, not the captioned {kind}")
    over_time = namespace["over_time"]
    days, day_positions = two_days(over_time)
    summary = over_time.summary_graph()
    if summary.kind != "digraph":
        raise ValueError(f"the collapsed graph is a {summary.kind}, not a cyclic graph")
    # The collapsed graph in one row, its variables in the days' order.
    p = len(over_time.labels)
    rows = {i: (i / max(1, p - 1), 0.5) for i in range(p)}
    drawn = [
        (
            graphs[key],
            f"The {kind} {name} returns",
            positions,
            f"<strong>{kind}</strong>, from {name}. {escape(what)}",
        )
        for key, (name, kind, what) in INTERFACE_CAPTIONS.items()
    ] + [
        (
            structure,
            f"The {kind.lower()} VARMA-LiNGAM returns",
            where,
            f"<strong>{kind}</strong>, {escape(what)}",
        )
        for (structure, where), (kind, what) in zip(
            [(days, day_positions), (summary, rows)], TEMPORAL_CAPTIONS, strict=True
        )
    ]
    figures = "".join(
        f'<figure class="interface-figure">{drawing(structure, label, where, size=(300, 210))}'
        f"<figcaption>{caption}</figcaption></figure>"
        for structure, label, where, caption in drawn
    )
    return shown, printed, figures


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


def agent_demo(frames: list[tuple[str, list[str], int]], figure: str) -> str:
    """The agent's three steps in a row, then its session in a terminal beside the graph it
    returned.

    Every line is in the page; the page script only reveals them in turn, and without it, or with
    reduced motion, the session is shown whole.
    """
    count = len(frames[0][1]) + frames[0][2]
    steps = (
        (
            "Find a method",
            f"<code>andrey list</code> names all {count} methods, with the graph each returns.",
        ),
        (
            "Read its contract",
            "<code>andrey run pc --help --json</code> gives each parameter's type, "
            "default, and meaning as JSON Schema.",
        ),
        ("Run it", "The run prints the graph as JSON, its nodes named by the data's header row."),
    )
    shown = []
    for command, lines, more in frames:
        lexer = "json" if lines[0].startswith(("{", "[")) else "text"
        body = [
            f'<span class="ln" style="--i: {i}">{code_html(line, lexer)}</span>'
            for i, line in enumerate(lines)
        ]
        if more:
            body.append(f'<span class="ln elide" style="--i: {len(lines)}">... {more} more</span>')
        shown.append(
            '<div class="frame"><div class="cmd"><span class="prompt">$</span> '
            f'<span class="cmd-text">{escape(command)}</span></div>'
            f'<div class="out hl">{"".join(body)}</div><span class="exit">exit 0</span></div>'
        )
    return (
        '<div class="agent-demo"><ol class="agent-steps">'
        + "".join(f'<li class="agent-step"><b>{t}</b><p>{text}</p></li>' for t, text in steps)
        + '</ol><div class="agent-run"><div class="term"><div class="term-bar window-bar">'
        '<button class="term-replay" type="button" hidden>Replay'
        '</button></div><div class="term-body" tabindex="0" role="region" '
        f'aria-label="Agent session">{"".join(shown)}</div></div><figure class="agent-graph">'
        f"{figure}<figcaption>The graph in that JSON, drawn with <code>andrey.viz.draw</code>."
        "</figcaption></figure></div></div>"
    )


# Icons from Feather (MIT, https://github.com/feathericons/feather), and the robot from Lucide (ISC,
# https://github.com/lucide-icons/lucide), drawn at one size and stroke.
ICON = {
    "terminal": '<polyline points="4 17 10 11 4 5"/><line x1="12" y1="19" x2="20" y2="19"/>',
    "bot": (
        '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/>'
        '<path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>'
    ),
    "chart": (
        '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/>'
        '<line x1="6" y1="20" x2="6" y2="14"/>'
    ),
    "graph": (
        '<circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/>'
        '<circle cx="18" cy="19" r="3"/>'
        '<line x1="8.59" y1="13.51" x2="15.42" y2="17.49"/>'
        '<line x1="15.41" y1="6.51" x2="8.59" y2="10.49"/>'
    ),
    "code": '<polyline points="16 18 22 12 16 6"/><polyline points="8 6 2 12 8 18"/>',
    "check": '<polyline points="20 6 9 17 4 12"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/>'
    '<circle cx="12" cy="12" r="2"/>',
    "minus": '<line x1="5" y1="12" x2="19" y2="12"/>',
    "half": '<circle cx="12" cy="12" r="8"/><path d="M12 4a8 8 0 0 0 0 16z" fill="currentColor"/>',
    "x": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
}
# The documentation pages a first visit wants, in the order the band lists them.
DOC_PAGES = (
    (
        "terminal",
        "docs/guides/getting-started.html",
        "Getting started",
        "Install, then recover your first graph.",
    ),
    (
        "graph",
        "docs/guides/graph-types.html",
        "Graph types",
        "What a DAG, CPDAG, PAG, or temporal graph claims.",
    ),
    ("code", "docs/api.html", "API reference", "Every method, structure, and command."),
    (
        "chart",
        "docs/benchmarks.html",
        "Release cards",
        "Timings, accuracy, limits, and how to rerun them.",
    ),
)


def glyph(icon: str) -> str:
    """One line symbol, inheriting the color of the text beside it."""
    return (
        '<svg class="icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{ICON[icon]}</svg>'
    )


def chip(icon: str, url: str, text: str, *, primary: bool = False) -> str:
    """A link with its symbol, boxed; the hero uses one treatment for all of them."""
    kind = "chip chip--primary" if primary else "chip"
    return f'<a class="{kind}" href="{escape(url)}">{glyph(icon)}{escape(text)}</a>'


# The hero's art (site/brand/art-<name>.jpg), which the first section's title shares, the art of
# "From data to graph" and of the documentation, and each bento tile's (site/brand/bento/
# <name>.jpg): a crop of a cover's art at twice the tile's size.
HERO_ART = "lanes"
EXAMPLE_ART = "start"
DOCS_ART = "docs"
TILE_SIZES = {"4": (768, 416), "6": (1168, 416)}
PRIORITY = ' fetchpriority="high"'


def picture(path: str, attrs: str) -> str:
    """A brand JPEG (``_static/<path>.jpg``) with its AVIF copy first, which a browser that reads
    AVIF loads instead (``docs/conf.py`` writes the copies); ``attrs`` are the image's."""
    return (
        f'<picture><source type="image/avif" srcset="_static/{path}.avif">'
        f'<img src="_static/{path}.jpg" {attrs}></picture>'
    )


def cover_frame(art: str, *, priority: bool = False) -> str:
    """An art in a frame of its own proportions, with the sweep where its art has one, so the
    sweep lies on the art at every crop. Only the first screen's art loads before the rest."""
    load = PRIORITY if priority else ' loading="lazy"'
    return (
        '<div class="cover-frame">'
        + picture(
            f"art-{art}",
            f'alt="{escape(cover_alt(art), quote=True)}" width="1600" height="640"{load}',
        )
        + f"{sweep(art) if art in SWEPT else ''}</div>"
    )


def hero_art() -> str:
    """The hero's art, the fast lanes: it stays on the screen while the hero's type scrolls away
    and the first section's title comes up over it."""
    return cover_frame(HERO_ART, priority=True)


def tile_art(name: str, size: str) -> str:
    """A tile's art, behind its text: a crop of the art of the same name."""
    width, height = TILE_SIZES[size]
    return picture(
        f"bento/{name}",
        f'class="tile-art" alt="{escape(cover_alt(name), quote=True)}" width="{width}" '
        f'height="{height}" loading="lazy"',
    )


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
    """The pill above the headline: it copies ``AGENT_PROMPT`` to paste into a coding agent
    (``prompt.js``), and without the script it leads to the FAQ answer that gives the prompt."""
    logos = "".join(agent_logo(name) for name in AGENTS)
    return (
        f'<a class="agent-pill" href="docs/faq.html#how-does-an-agent-use-it" '
        f'data-prompt="{escape(AGENT_PROMPT)}" title="Copy a prompt for your coding agent">'
        '<span class="agent-pill__label">Onboard your agent<span class="agent-pill__more"> to '
        'Andrey</span></span><span class="agent-logos" aria-hidden="true">'
        f"{logos}</span></a>"
    )


def hero_chips() -> str:
    """The buttons beside the install line: Get started, and the live demo once its Space is
    public."""
    start = chip("terminal", "docs/guides/getting-started.html", "Get started", primary=True)
    return start + (chip("graph", DEMO_URL, "Live demo") if DEMO_URL else "")


def docs_grid() -> str:
    """The documentation band's four entries, each a page with one line about it."""
    return "".join(
        f'<li><a class="doc-link" href="{escape(url)}">{glyph(icon)}{escape(text)}</a>'
        f"<span>{escape(blurb)}</span></li>"
        for icon, url, text, blurb in DOC_PAGES
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


def plus() -> str:
    """The round plus a tile carries when it opens more than it shows."""
    return f'<span class="wf-plus" aria-hidden="true">{_why_fast["PLUS"]}</span>'


# The page's four sections, each a word of the headline: its id, title, cover art (site/brand/
# art-*.jpg), and the picture its card in the hero shows, with its size: a tile's crop
# (site/brand/bento/*.jpg), or the cover itself where the crop is cobalt only and would lose the
# art's amber.
ACTS = (
    ("accelerated", "Accelerated", "lanes", ("bento/lanes.jpg", TILE_SIZES["6"])),
    ("accurate", "Accurate", "contour", ("art-contour.jpg", (1600, 640))),
    ("agentic", "Agentic", "agentic", ("bento/agentic.jpg", TILE_SIZES["6"])),
    ("all-in-one", "All-in-one", "methods", ("bento/methods.jpg", TILE_SIZES["6"])),
)


def act_claims(featured: dict[str, dict]) -> dict[str, tuple[str, str]]:
    """Each section's claim, from the summaries and the registry: a short form for its card in the
    hero, and the sentence under its title. Speed is ``CLAIM``, with no number. Accuracy is said
    of the featured methods (:func:`featured_methods`), the ones its figure shows, and the build
    stops when one has more errors than a package it is compared with there."""
    worse = sorted(
        {e["name"] for e in featured.values() for _, r in e["cells"] if outcome(r) in WORSE}
    )
    if worse:
        raise ValueError(f"the Accurate claim does not hold for {', '.join(worse)}")
    count = sum(spec.status == "supported" for spec in list_specs())
    kinds = NUMBER_WORDS[len(graph_kinds())]
    return {
        "accelerated": ("The fastest package", CLAIM.replace("fastest", "<b>fastest</b>")),
        "accurate": ("Just as accurate", "<b>Just as accurate</b> as other popular packages."),
        "agentic": (
            "Built for agents",
            "A command line that never prompts and answers in JSON, so an agent can run it alone.",
        ),
        "all-in-one": (
            f"{count} methods, {kinds} kinds of graph",
            f"<b>{count} supported methods</b> for every kind of data, and {kinds} kinds of causal"
            " graph.",
        ),
    }


def act_cards(claims: dict[str, tuple[str, str]]) -> str:
    """The hero's four cards, one per section: its art, its title, and its claim's short form."""
    return "".join(
        f'<li class="act-card"><a href="#{key}">'
        + picture(
            image.removesuffix(".jpg"),
            f'alt="{escape(cover_alt(art), quote=True)}" width="{width}" height="{height}"',
        )
        + f"<b>{title}</b><span>{escape(claims[key][0])}</span></a></li>"
        for key, title, art, (image, (width, height)) in ACTS
    )


# The covers whose amber streak, trail, arm, arrow, edge, or beam a trace of light sweeps along,
# bottom to tip, or a level beam from left to right (``streak``).
SWEPT = ("lanes", "contour", "agentic", "methods", "start", "docs")


def streak(art: str, points: int = 16) -> list[tuple[float, float]]:
    """The centerline of an art's amber streak, or its trail of amber dots, in its 1600 x 640
    frame from the bottom to the tip, or, for a level beam, from left to right, the way its light
    travels: the warm pixels cut into ``points`` equal bands along the streak's longer extent, and
    each band's median point, so the path follows a bend and runs through a dotted trail's dots.
    Fails when the art has no streak to follow."""
    from PIL import Image  # the docs group's; the unit tests import this module without it

    with Image.open(SITE / "brand" / f"art-{art}.jpg") as image:
        rgb = np.asarray(image.convert("RGB"), dtype=float)
    warm = (rgb[..., 0] - rgb[..., 2] > 40) & (rgb[..., 0] > 120)
    ys, xs = np.nonzero(warm)
    if len(xs) == 0 or max(np.ptp(xs), np.ptp(ys)) < min(warm.shape) / 2:
        raise ValueError(f"art-{art}.jpg has no amber streak to sweep along")
    along = ys if np.ptp(ys) > np.ptp(xs) else xs
    edges = np.linspace(along.min(), along.max(), points + 1)
    path = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        band = (along >= low) & (along <= high)
        if band.any():
            path.append((float(np.median(xs[band])), float(np.median(ys[band]))))
    if np.ptp(ys) < np.ptp(xs) / 8:
        return path  # level: the bands run left to right
    return path if path[0][1] > path[-1][1] else path[::-1]


def sweep(art: str) -> str:
    """The trace that sweeps along ``art``'s streak: a soft glow and a paler core on the same path,
    in the art's own frame, so it lies on the streak at every crop. The page script runs it while
    the cover is on the screen (``tiles.js``)."""
    d = "M" + " L".join(f"{x:.0f} {y:.0f}" for x, y in streak(art))
    return (
        '<svg class="cover-sweep" viewBox="0 0 1600 640" preserveAspectRatio="none" '
        'aria-hidden="true" focusable="false"><defs>'
        f'<filter id="sweep-{art}-glow" x="-10%" y="-10%" width="120%" height="120%">'
        '<feGaussianBlur stdDeviation="13"/></filter>'
        f'<filter id="sweep-{art}-core"><feGaussianBlur stdDeviation="2.4"/></filter></defs>'
        f'<path class="sweep-glow" d="{d}" pathLength="100" filter="url(#sweep-{art}-glow)"/>'
        f'<path class="sweep-core" d="{d}" pathLength="100" filter="url(#sweep-{art}-core)"/></svg>'
    )


def act_covers(claims: dict[str, tuple[str, str]]) -> dict[str, str]:
    """Each section's opening: its art at full bleed, its title, and its claim. The section whose
    art is the hero's opens on the hero's art instead, which stays on the screen into it."""
    covers = {}
    for key, title, art, _ in ACTS:
        head = (
            f'<div class="act-head"><h2 id="{key}-title" class="reveal">{title}</h2>'
            f'<p class="act-claim reveal">{claims[key][1]}</p></div>'
        )
        covers[key] = (
            f'<div class="act-cover act-cover--hero">{head}</div>'
            if art == HERO_ART
            else f'<div class="section-cover act-cover"><div class="cover-art" aria-hidden="true">'
            f"{cover_frame(art)}</div>{head}</div>"
        )
    return covers


def example_cover() -> str:
    """The opening of "From data to graph", inside the last section: its art, its title, and its
    line."""
    return (
        '<div class="section-cover act-cover act-cover--sub" id="data-to-graph">'
        f'<div class="cover-art" aria-hidden="true">{cover_frame(EXAMPLE_ART)}</div>'
        '<div class="act-head"><h3 class="reveal">From data to graph</h3>'
        '<p class="act-claim reveal">From a table of measurements to its causal graph, in a few '
        "lines of Python.</p></div></div>"
    )


def docs_cover() -> str:
    """The documentation's opening: its art and its title."""
    return (
        '<div class="section-cover act-cover">'
        f'<div class="cover-art" aria-hidden="true">{cover_frame(DOCS_ART)}</div>'
        '<div class="act-head"><h2 id="docs-title" class="reveal">Documentation</h2></div></div>'
    )


def glass_tile(span: str, title: str, text: str, glass: str, *, key: str = "") -> str:
    """A tile whose evidence sits on a glass card under its title. It opens the dialog ``key``
    when that holds more than the tile shows; without one, it is a box and carries no plus. The
    card is decorative where the dialog says it in full."""
    hidden = ' aria-hidden="true"' if key else ""
    body = (
        f'<h3 class="tile-title">{title}</h3>'
        + (f'<p class="tile-text">{text}</p>' if text else "")
        + f'<div class="tile-glass"{hidden}>{glass}</div>'
    )
    if not key:
        return f'<li class="tile tile--{span} tile--glass"><div class="tile-face">{body}</div></li>'
    return (
        f'<li class="tile tile--{span} tile--glass"><a class="tile-face" href="#{key}" '
        f'data-tile="{key}">{body}{plus()}</a></li>'
    )


def speed_tiles(featured: dict[str, dict]) -> str:
    """Accelerated's evidence: each featured method's fit time beside every package's
    (:func:`featured_bars`), and beside it where the speed comes from."""
    return glass_tile("7", "Fit time", "", featured_bars(featured, "s", seconds_text)) + (
        '<li class="bento-note"><h3>Why Andrey is fast</h3>'
        "<p>It does many things at once, never computes twice, skips what can't matter, and "
        "uses the hardware you have.</p>"
        '<p><a href="blog/20261008-why-andrey-is-fast.html">Explore why Andrey is so fast</a>'
        "</p></li>"
    )


def accuracy_tiles(featured: dict[str, dict]) -> str:
    """Accurate's evidence: each featured method's errors beside every package's
    (:func:`featured_bars`), and beside it where every method's errors are: the release cards."""
    return glass_tile(
        "7", "Errors against the true graph", "", featured_bars(featured, "errors", number_text)
    ) + (
        '<li class="bento-note"><h3>Tested against the true graphs</h3>'
        "<p>Every release is tested against the graphs that generated the data.</p>"
        '<p><a href="docs/benchmarks.html">Release cards</a></p></li>'
    )


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
    return f'<pre class="tile-term">{chr(10).join(lines)}</pre>'


def agent_tile(frames: list[tuple[str, list[str], int]]) -> str:
    """An agent's session at the command line, from the list of methods to the graph, as one tile
    across the row. It opens the agents dialog, which plays the session step by step beside the
    graph it returns."""
    return glass_tile(
        "wide",
        "Find a method, read it, run it",
        "",
        terminal(frames, shown=4),
        key="agents",
    )


def names_only(figures: str) -> str:
    """The graph figures with each caption cut to its method's name."""
    return re.sub(
        r"<figcaption><strong>(.*?)</strong>.*?</figcaption>",
        r"<figcaption>\1</figcaption>",
        figures,
        flags=re.S,
    )


def coverage_tiles(graphs: str) -> str:
    """Andrey beside other packages (:func:`comparison_table`), its supported methods by family
    and kind of data (:func:`families_table`), and the kinds of causal graph its methods
    return."""
    return (
        glass_tile(
            "wide",
            "Package comparison",
            '<a href="docs/comparison.html">Full package comparison</a>',
            comparison_table(),
        )
        + glass_tile(
            "wide",
            "Methods by kind of data",
            "",
            f'<div class="matrix-card">{families_table()}</div>',
        )
        + glass_tile(
            "wide",
            graph_title(),
            "",
            f'<div class="glass-graphs">{names_only(graphs)}</div>',
            key="graphs",
        )
    )


def tile(key: str, size: str, body: str, *, kind: str = "") -> str:
    """One bento tile: a link that opens the dialog ``key`` and, without script, jumps to the same
    id, where the dialog's content is shown in the page."""
    classes = f"tile tile--{size}" + (f" tile--{kind}" if kind else "")
    return (
        f'<li class="{classes}"><a class="tile-face" href="#{key}" data-tile="{key}">{body}{plus()}'
        "</a></li>"
    )


def dialog_head(key: str, title: str, intro: str = "") -> str:
    """The start of a tile's dialog, its heading and close button; ``DIALOG_TAIL`` ends it."""
    return (
        f'<dialog class="wf tile-dialog" id="{key}" aria-labelledby="{key}-title">'
        f'<div class="wf-top"><div><h2 id="{key}-title">{escape(title)}</h2>'
        f"{f'<p>{intro}</p>' if intro else ''}</div>"
        f'<button class="wf-close" type="button" aria-label="Close">{_why_fast["CLOSE"]}'
        '</button></div><div class="tile-body">'
    )


DIALOG_TAIL = "</div></dialog>"


def package_name(comparator: str) -> str:
    """The package ``comparator`` names, without a version or setting, from ``PACKAGE_NAMES``;
    a name that still holds a digit raises ``ValueError``."""
    name = PACKAGE_NAMES.get(comparator, comparator)
    if re.search(r"\d", name):
        raise ValueError(f"name the package of {comparator!r} in PACKAGE_NAMES")
    return name


# ---- compared with other packages ------------------------------------------------------------
# Andrey compared with other packages. Feature marks come from released code
# (causal-learn 0.1.4.8, Tetrad and causal-cmd 7.6.10, pcalg 2.7-12, bnlearn 5.2.1). Hooks for
# custom tests or scores do not count as implementations. Features only on a development branch
# do not count as released. Permutation invariance uses the 0.7.0 post's measurements; open
# benchmarks use each package's README, documentation, and release notes. A real-data feature is
# a yes when the package's main learners have it, and partly when only some do. Interventional
# data is a yes when a method takes each row's environment or the intervention targets.
PEERS = ("Andrey", "causal-learn", "Tetrad", "pcalg", "bnlearn")
PEER_VERSIONS = "causal-learn 0.1.4.8, Tetrad 7.6.10, pcalg 2.7-12, and bnlearn 5.2.1"
# Each mark's icon on the homepage, its words for screen readers, and its README symbol.
MARKS = {
    "yes": ("check", "Yes", "\u2713"),
    "part": ("half", "Partly", "\u25d0"),
    "no": ("x", "No", "\u2717"),
}
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
# registry: gies runs ges's search, multi_group_direct_lingam extends direct_lingam, inter_iamb and
# fast_iamb are variants of iamb, and mmhc is rsmax2 with its default phases, as bnlearn's are
# counted. The others are read from each package's released code.
COUNTED_WITH = {
    "gies": "ges",
    "multi_group_direct_lingam": "direct_lingam",
    "inter_iamb": "iamb",
    "fast_iamb": "iamb",
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
    """Measured rows as (section, label, values in PEERS order).

    Speed has no pooled value across methods; the speed figures compare each method separately.
    """
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


def alone(marks: tuple[str, ...]) -> bool:
    """Whether only Andrey, the first of ``PEERS``, has a feature in full."""
    return marks[0] == "yes" and "yes" not in marks[1:]


def comparison_table() -> str:
    """The homepage's comparison, a column per package in ``PEERS``: the measured rows
    (:func:`measured_rows`), then the features only Andrey has in full (:func:`alone`), by section.
    The comparison page (:func:`comparison_page`) has every feature and the notes; here each
    partial mark keeps its note as text for screen readers and as a tooltip."""
    notes = {(f, p): note for f, p, _, note in comparison_notes()}
    head = "".join(
        f'<th scope="col"{" class=ct-ours" if p == "Andrey" else ""}>{escape(p)}</th>'
        for p in PEERS
    )
    measured = measured_rows()
    rows = []
    for section in SECTIONS:
        group = [r for r in measured if r[0] == section]
        features = [r for r in COMPARISON if r[0] == section and alone(r[2])]
        if not group and not features:
            continue
        rows.append(
            f'<tr class="compare-group"><th scope="colgroup" colspan="{1 + len(PEERS)}">'
            f"{section}</th></tr>"
        )
        for _, label, values in group:
            cells = "".join(
                f'<td class="value{" ct-ours" if p == "Andrey" else ""}">{v}</td>'
                for p, v in zip(PEERS, values, strict=True)
            )
            name = escape(label)
            if label == "Distinct methods":
                name = f'<a href="{CATALOG}">{name}</a>'
            rows.append(f'<tr class="compare-alone"><th scope="row">{name}</th>{cells}</tr>')
        for _, feature, marks, _ in features:
            cells = []
            for package, mark in zip(PEERS, marks, strict=True):
                icon, word, _ = MARKS[mark]
                note = notes.get((feature, package))
                said = f"{word}: {escape(note)}" if note else word
                tip = f' title="{escape(note)}"' if note else ""
                ours = " ct-ours" if package == "Andrey" else ""
                cells.append(
                    f'<td class="mark mark--{mark}{ours}"{tip}>{glyph(icon)}'
                    f'<span class="visually-hidden">{said}</span></td>'
                )
            # Only Andrey has it, so its mark is the row's one amber.
            rows.append(
                f'<tr class="compare-alone"><th scope="row">{escape(feature)}</th>'
                f"{''.join(cells)}</tr>"
            )
    return (
        '<p class="compare-hint">Scroll to see all packages.</p>'
        '<div class="table-scroll" role="region" aria-label="Feature comparison" tabindex="0">'
        '<table class="compare">'
        f"<thead><tr><td></td>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"
    )


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
                cells.append(f"<td{ours if package == 'Andrey' else ''}>{MARKS[mark][1]}{ref}</td>")
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


def homepage_context(summary: dict, *, placeholder: bool = False) -> dict[str, str]:
    """Everything the homepage template shows, as HTML, from the first release's summary and the
    later releases' published ones."""
    fastest(summary)
    summaries = release_summaries(summary)
    code, printed, figure = run_example()
    _, _, interface_figures = run_interface()
    featured = featured_methods(summaries)
    claims = act_claims(featured)
    frames = agent_frames()
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
        "agent_pill": agent_pill(),
        # The first screen's art, which the page asks for before its stylesheets load.
        "hero_image": f"_static/art-{HERO_ART}.avif",
        "hero_chips": hero_chips(),
        "hero_art": hero_art(),
        "tile_art": tile_art,
        "plus": plus(),
        "act_cards": act_cards(claims),
        "act_covers": act_covers(claims),
        "example_cover": example_cover(),
        "docs_cover": docs_cover(),
        "speed_tiles": speed_tiles(featured),
        "accuracy_tiles": accuracy_tiles(featured),
        "agent_tile": agent_tile(frames),
        "coverage_tiles": coverage_tiles(interface_figures),
        "dialog_head": dialog_head,
        "dialog_tail": DIALOG_TAIL,
        "method_count": str(len(list_specs())),
        "graph_title": graph_title(),
        "interface_figures": interface_figures,
        "example_code": code_html(code.rstrip(), "python"),
        "example_output": escape(printed),
        "example_figure": figure,
        "agent_demo": agent_demo(frames, figure),
        "docs_grid": docs_grid(),
        "style": homepage_code_css(),
    }


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


# ---- the featured methods' figures ------------------------------------------------------------
# The outcomes :func:`outcome` gives where Andrey has more errors than the other package.
WORSE = {"worse", "more"}


def outcome(row: dict) -> str:
    """The outcome a row records for Andrey against the other package: its release card's verdict
    (``card_verdict``: "tie" where the paired 95% interval of the difference in errors contains 0),
    or where the card records none, the two error counts on the same datasets: "same", "fewer",
    or "more"."""
    if row["card_verdict"] in ("tie", "better", "worse"):
        return row["card_verdict"]
    ours, theirs = row["andrey_errors"], row["comparator_errors"]
    return "same" if abs(ours - theirs) < 1e-9 else "fewer" if ours < theirs else "more"


def featured_bars(featured: dict[str, dict], measure: str, text) -> str:  # noqa: ANN001
    """A row per featured method (:func:`featured_methods`) at the size it is shown: Andrey's bar
    in the accent, then each other package's, on the row's own scale from zero, each labeled with
    its package and its value in ``text``. ``measure`` names the rows' fields: ``s`` for the fit
    time, ``errors`` for the errors against the true graph."""
    rows = []
    for entry in featured.values():
        cells = entry["cells"]
        values = [("Andrey", cells[0][1][f"andrey_{measure}"])]
        values += [(package_name(m["comparator"]), r[f"comparator_{measure}"]) for m, r in cells]
        top = max(value for _, value in values) or 1
        bars = "".join(
            f'<div class="err-bar{" err-bar--andrey" if k == 0 else ""}"><span>{escape(who)}'
            f'</span><i style="--f: {value / top:.4f}"></i><b>{escape(text(value))}</b></div>'
            for k, (who, value) in enumerate(values)
        )
        rows.append(
            f'<li><p class="err-head"><b>{escape(entry["name"])}</b>'
            f"<span>{entry['size']:,} {escape(entry['unit'])}</span></p>{bars}</li>"
        )
    return f'<ol class="err-pairs">{"".join(rows)}</ol>'


# ---- the families, by the kinds of data their methods read -----------------------------------
CATALOG = "docs/methods.html"
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
    "Directed graph": "directed-cycles-the-kind-matters",
}


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
    page = DOCS / "docs" / "benchmarks" / f"{release}.md"
    return f"benchmarks/{release}.html" if page.exists() else "benchmarks.html"


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
    return table_html(["Data", "Method", "Graph", "Families"], rows) + "\n"


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
            + table_html(["Method", "Data", "Graph", "Compared with"], rows),
            "```",
            "",
        ]
    return "\n".join(out)

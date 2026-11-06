"""The environment and target arguments of ``gies`` and ``cdnod``, in Python and on the CLI.

Every accepted form of ``env`` and ``targets`` gives the same fit as the 0/1 intervention matrix
it describes, each former form warns once and gives the same fit as its replacement, and every
form that cannot say which variables an environment intervened on raises. Data given as each
environment's rows apart fit, in every method that takes ``env``, what the table they stack into
fits.
"""

from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import andrey
from andrey.cli import main
from andrey.constraint.cdnod import cdnod as cdnod_engine
from andrey.resample import build_job
from andrey.spec import get_spec

N = 900
#: Each row's environment: observational rows, then rows that set x1, then rows that set x2.
ENV = np.repeat(["observed", "set x1", "set x2"], [500, 200, 200])
TARGETS = {"observed": [], "set x1": ["x1"], "set x2": ["x2"]}


def _chain() -> pd.DataFrame:
    """A chain x0 -> x1 -> x2 whose rows of ``set x1`` and ``set x2`` set that variable."""
    rng = np.random.default_rng(0)
    x0 = rng.standard_normal(N)
    x1 = 0.8 * x0 + rng.standard_normal(N)
    x1[ENV == "set x1"] = rng.standard_normal(200)
    x2 = 0.8 * x1 + rng.standard_normal(N)
    x2[ENV == "set x2"] = rng.standard_normal(200)
    return pd.DataFrame({"x0": x0, "x1": x1, "x2": x2})


DATA = _chain()
MASK = np.column_stack([np.zeros(N, bool), ENV == "set x1", ENV == "set x2"])


def _same(a: andrey.StructureOutput, b: andrey.StructureOutput) -> bool:
    """Whether two fits return the same graph, labels, and metadata, the score bit for bit."""
    score = [np.float64(o.metadata.get("score", 0.0)).tobytes() for o in (a, b)]
    return (
        np.array_equal(a.structure.to_numpy(), b.structure.to_numpy())
        and a.structure.labels == b.structure.labels
        and a.metadata == b.metadata
        and score[0] == score[1]
    )


@pytest.fixture(scope="module")
def reference() -> andrey.StructureOutput:
    return andrey.gies(DATA, targets=MASK)


def test_the_reference_fit_orients_the_chain(reference):
    assert reference.structure.oriented_edges() == [
        ("x0", "x1", "directed"),
        ("x1", "x2", "directed"),
    ]


# --- gies: env and targets as columns of the data -------------------------------------------------


GIES_FORMS = {
    "env column": lambda: andrey.gies(DATA.assign(e=ENV), "e", targets=TARGETS),
    "targets column": lambda: andrey.gies(
        DATA.assign(t=np.select([ENV == "set x1", ENV == "set x2"], ["x1", "x2"], "")),
        targets="t",
    ),
}


@pytest.mark.parametrize("form", list(GIES_FORMS))
def test_each_form_of_targets_fits_what_its_matrix_fits(form, reference):
    with warnings.catch_warnings():
        warnings.simplefilter("error", andrey.DeprecatedArgumentWarning)
        assert _same(GIES_FORMS[form](), reference)


def test_a_text_column_of_env_or_targets_leaves_the_default_score():
    """The columns leave the data before the score is picked from the column types."""
    cells = pd.Categorical(np.select([ENV == "set x1", ENV == "set x2"], ["x1", "x2"], ""))
    out = andrey.gies(DATA.assign(e=ENV, t=cells), "e", targets="t")
    assert out.metadata["score_func"] == "bic" and out.structure.n_nodes == 3


def test_one_observational_environment_is_observational_data():
    none = andrey.gies(DATA, targets=np.zeros((N, 3)))
    assert _same(andrey.gies(DATA, np.zeros(N), targets={0: []}), none)
    assert none.structure == andrey.gies(DATA).structure


def test_the_matrix_as_the_second_argument_is_refused_with_its_place_now():
    with pytest.raises(ValueError, match=r"matrix .* goes in targets since 0\.14\.0"):
        andrey.gies(DATA, MASK)


GIES_ERRORS = {
    "env without targets": lambda: andrey.gies(DATA, ENV),
    "a mapping without env": lambda: andrey.gies(DATA, targets=TARGETS),
    "a key env lacks": lambda: andrey.gies(DATA, ENV, targets={**TARGETS, "set x3": ["x0"]}),
    "an environment the mapping leaves out": lambda: andrey.gies(
        DATA, ENV, targets={"observed": [], "set x1": ["x1"]}
    ),
    "unknown targets": lambda: andrey.gies(DATA, ENV, targets={**TARGETS, "set x2": None}),
    "a variable data lacks": lambda: andrey.gies(DATA, ENV, targets={**TARGETS, "set x2": ["x9"]}),
    "a position data lacks": lambda: andrey.gies(DATA, ENV, targets={**TARGETS, "set x2": [3]}),
    "a variable that is not a name or a position": lambda: andrey.gies(
        DATA, ENV, targets={**TARGETS, "set x2": [2.0]}
    ),
    "a matrix that splits an environment": lambda: andrey.gies(
        DATA, np.where(ENV == "set x2", "set x1", ENV), targets=MASK
    ),
    "a column that splits an environment": lambda: andrey.gies(
        DATA.assign(t=np.select([ENV == "set x1"], ["x1"], "")), np.zeros(N), targets="t"
    ),
    "env of another length": lambda: andrey.gies(DATA, ENV[1:], targets=TARGETS),
    "env with a missing label": lambda: andrey.gies(DATA, np.r_[None, ENV[1:]], targets=TARGETS),
    "env with a NaN label": lambda: andrey.gies(
        DATA, np.r_[np.nan, np.zeros(N - 1)], targets={0.0: []}
    ),
    "env that mixes strings and numbers": lambda: andrey.gies(
        DATA, np.r_[np.array([0], dtype=object), ENV[1:]], targets={**TARGETS, 0: []}
    ),
    "env that names a column data lacks": lambda: andrey.gies(DATA, "e", targets=TARGETS),
    "env that names a column of an array": lambda: andrey.gies(
        DATA.to_numpy(), "e", targets=TARGETS
    ),
    "env and targets that name one column": lambda: andrey.gies(
        DATA.assign(e=ENV), "e", targets="e"
    ),
    "the matrix as the second argument and targets": lambda: andrey.gies(
        DATA, MASK, targets=TARGETS
    ),
}


@pytest.mark.parametrize("case", list(GIES_ERRORS))
def test_gies_refuses_targets_it_cannot_read(case):
    with pytest.raises(ValueError):
        GIES_ERRORS[case]()


def test_bootstrap_draws_within_the_environments_of_env_in_every_form():
    """A resample takes the same share of each environment's rows, whatever form names them."""
    fits = [
        andrey.bootstrap("gies", DATA, n_resamples=2, env=ENV, targets=TARGETS),
        andrey.bootstrap("gies", DATA.assign(e=ENV), n_resamples=2, env="e", targets=TARGETS),
        andrey.bootstrap("gies", DATA, n_resamples=2, env=ENV, targets=MASK),
    ]
    assert all(np.array_equal(f.frequency("->"), fits[0].frequency("->")) for f in fits)


def test_a_bootstrap_resample_keeps_every_environment():
    """An environment of one row is in every subsample, so its key in targets always has rows."""
    env = ENV.copy()
    env[0] = "one row"
    out = andrey.bootstrap(
        "gies", DATA, n_resamples=20, env=env, targets={**TARGETS, "one row": []}
    )
    assert out.n_resamples == 20


# --- cdnod: env as numbers, labels, and a column -----------------------------------------------


def _domains() -> tuple[np.ndarray, np.ndarray]:
    """Two variables whose cause's mean shifts with the domain, and the domain of each row."""
    rng = np.random.default_rng(1)
    domain = np.repeat([0.0, 1.0, 2.0], 200)
    x = domain + rng.standard_normal(600)
    return np.column_stack([x, x + 0.5 * rng.standard_normal(600)]), domain


X, DOMAIN = _domains()


def test_cdnod_reads_numbers_as_numbers():
    """Numbers reach the search unchanged, in any order and of any spacing."""
    for index in (DOMAIN, 2.0 - DOMAIN, np.sqrt(DOMAIN) + 0.25, np.arange(600) / 7.0):
        out = andrey.cdnod(X, index)
        engine = cdnod_engine(X, index)
        assert np.array_equal(out.structure.to_numpy(), engine.to_numpy())


def test_cdnod_codes_text_labels_in_sorted_order():
    labels = np.array(["b", "c", "a"])[DOMAIN.astype(int)]
    codes = np.array([1.0, 2.0, 0.0])[DOMAIN.astype(int)]
    assert _same(andrey.cdnod(X, labels), andrey.cdnod(X, codes))


def test_reordering_the_rows_leaves_cdnod_on_text_labels_unchanged():
    """With three domains and fisherz, a shift in the middle domain alone keeps the index apart
    from x under the sorted codes; codes in order of first appearance would put that domain first
    whenever its rows came first, and orient x -> y."""
    rng = np.random.default_rng(2)
    labels = np.repeat(["a", "b", "c"], 300)
    x = np.where(labels == "b", 2.0, 0.0) + rng.standard_normal(900)
    data = np.column_stack([x, x + 0.5 * rng.standard_normal(900)])
    found = andrey.cdnod(data, labels, indep_test="fisherz").structure.to_numpy()
    for seed in range(3):
        order = np.random.default_rng(seed).permutation(900)
        order = np.r_[order[labels[order] == "b"], order[labels[order] != "b"]]  # "b" rows first
        reordered = andrey.cdnod(data[order], labels[order], indep_test="fisherz")
        assert np.array_equal(reordered.structure.to_numpy(), found)
    shifted_first = np.where(labels == "b", 0.0, np.where(labels == "a", 1.0, 2.0))
    assert not np.array_equal(andrey.cdnod(data, shifted_first).structure.to_numpy(), found)


CDNOD_FORMS = {
    "keyword": lambda: andrey.cdnod(X, env=DOMAIN),
    "column": lambda: andrey.cdnod(pd.DataFrame(X).assign(site=DOMAIN), "site"),
    "(n, 1) array": lambda: andrey.cdnod(X, DOMAIN.reshape(-1, 1)),
}


@pytest.mark.parametrize("form", list(CDNOD_FORMS))
def test_each_form_of_env_fits_what_its_numbers_fit(form):
    assert _same(CDNOD_FORMS[form](), andrey.cdnod(X, DOMAIN))


@pytest.mark.parametrize(
    "env",
    [
        pytest.param(np.r_[np.array(["a"], dtype=object), DOMAIN[1:]], id="strings and numbers"),
        pytest.param(np.r_[np.array([None], dtype=object), np.repeat("a", 599)], id="missing"),
        pytest.param("site", id="a column data lacks"),
    ],
)
def test_cdnod_refuses_env_it_cannot_read(env):
    with pytest.raises(ValueError):
        andrey.cdnod(pd.DataFrame(X), env)


def test_cdnod_takes_no_targets():
    with pytest.raises(TypeError):
        andrey.cdnod(X, DOMAIN, targets={0.0: []})


def test_bootstrap_carries_a_column_of_env_with_its_rows():
    frame = pd.DataFrame(X, columns=["x", "y"])
    by_column = andrey.bootstrap("cdnod", frame.assign(site=DOMAIN), n_resamples=2, env="site")
    by_values = andrey.bootstrap("cdnod", frame, n_resamples=2, env=DOMAIN)
    assert by_column.structure.labels == ("x", "y")
    assert np.array_equal(by_column.frequency("->"), by_values.frequency("->"))


def test_bootstrap_draws_cdnod_rows_within_domains_for_text():
    frame = pd.DataFrame(X, columns=["x", "y"])
    one_row = np.array(["a", "b", "c"])[DOMAIN.astype(int)]
    one_row[0] = "d"  # a domain of one row stays in every subsample
    assert andrey.bootstrap("cdnod", frame, n_resamples=4, env=one_row).n_resamples == 4


# --- data as each environment's rows apart -------------------------------------------------------

#: The rows of ``DATA`` by environment; ``ENV`` holds its labels in this order, so the tables stack
#: back into ``DATA``.
APART = {label: DATA[ENV == label] for label in TARGETS}
#: Each row's position in the list of ``APART``'s tables, the label a list gives it.
POSITIONS = np.repeat([0, 1, 2], [500, 200, 200])
#: ``ut_igsp``'s targets: ``set x2``'s unknown.
UT_TARGETS = {"observed": [], "set x1": ["x1"], "set x2": None}
#: The domains of ``X`` apart, in the order of ``DOMAIN``'s rows.
DOMAINS = [X[DOMAIN == k] for k in (0.0, 1.0, 2.0)]


def _with_nan(table):
    """A copy of ``table`` with a missing value at row 3, column 1."""
    table = table.copy()
    if hasattr(table, "iloc"):
        table.iloc[3, 1] = np.nan
    else:
        table[3, 1] = np.nan
    return table


#: Each call with the environments apart, and the call with the table they stack into.
APART_FITS = {
    "gies, a mapping of DataFrames": (
        lambda: andrey.gies(APART, targets=TARGETS),
        lambda: andrey.gies(DATA, ENV, targets=TARGETS),
    ),
    "gies, a list of arrays": (
        lambda: andrey.gies([t.to_numpy() for t in APART.values()], targets={0: [], 1: [1], 2: 2}),
        lambda: andrey.gies(DATA.to_numpy(), POSITIONS, targets={0: [], 1: [1], 2: 2}),
    ),
    "gies, DataFrames with their columns in another order": (
        lambda: andrey.gies(
            [t[["x2", "x0", "x1"]] if i else t for i, t in enumerate(APART.values())],
            targets={0: [], 1: ["x1"], 2: ["x2"]},
        ),
        lambda: andrey.gies(DATA, POSITIONS, targets={0: [], 1: ["x1"], 2: ["x2"]}),
    ),
    "gnies, a list of DataFrames": (
        lambda: andrey.gnies(list(APART.values())),
        lambda: andrey.gnies(DATA, POSITIONS),
    ),
    "gnies, a mapping with known targets": (
        lambda: andrey.gnies(APART, targets={"observed": []}),
        lambda: andrey.gnies(DATA, ENV, targets={"observed": []}),
    ),
    "ut_igsp, a mapping of DataFrames": (
        lambda: andrey.ut_igsp(APART, targets=UT_TARGETS),
        lambda: andrey.ut_igsp(DATA, ENV, targets=UT_TARGETS),
    ),
    "ut_igsp, a list of DataFrames": (
        lambda: andrey.ut_igsp(list(APART.values()), targets={0: [], 1: ["x1"], 2: None}),
        lambda: andrey.ut_igsp(DATA, POSITIONS, targets={0: [], 1: ["x1"], 2: None}),
    ),
    "cdnod, a list of arrays": (
        lambda: andrey.cdnod(DOMAINS),
        lambda: andrey.cdnod(X, DOMAIN),
    ),
    "cdnod, a list of arrays with a missing value, which cdnod accepts": (
        lambda: andrey.cdnod([DOMAINS[0], _with_nan(DOMAINS[1]), DOMAINS[2]]),
        lambda: andrey.cdnod(
            np.concatenate([DOMAINS[0], _with_nan(DOMAINS[1]), DOMAINS[2]]),
            np.repeat([0, 1, 2], [len(t) for t in DOMAINS]),
        ),
    ),
    "cdnod, a mapping of DataFrames": (
        lambda: andrey.cdnod(
            {k: pd.DataFrame(t, columns=["x", "y"]) for k, t in zip("abc", DOMAINS)}
        ),
        lambda: andrey.cdnod(
            pd.DataFrame(X, columns=["x", "y"]), np.array(["a", "b", "c"])[DOMAIN.astype(int)]
        ),
    ),
    "boss, a mapping of DataFrames": (
        lambda: andrey.boss(APART),
        lambda: andrey.boss(DATA, ENV),
    ),
    "grasp, a list of arrays": (
        lambda: andrey.grasp([t.to_numpy() for t in APART.values()]),
        lambda: andrey.grasp(DATA.to_numpy(), POSITIONS),
    ),
    "boss_fci, a list of DataFrames": (
        lambda: andrey.boss_fci(list(APART.values())),
        lambda: andrey.boss_fci(DATA, POSITIONS),
    ),
    "grasp_fci, a mapping of DataFrames": (
        lambda: andrey.grasp_fci(APART, confounded_index=False),
        lambda: andrey.grasp_fci(DATA, ENV, confounded_index=False),
    ),
    "ges, a mapping of DataFrames": (
        lambda: andrey.ges(APART),
        lambda: andrey.ges(DATA, ENV),
    ),
    "hc, a list of arrays": (
        lambda: andrey.hc([t.to_numpy() for t in APART.values()]),
        lambda: andrey.hc(DATA.to_numpy(), POSITIONS),
    ),
    "tabu, a list of DataFrames": (
        lambda: andrey.tabu(list(APART.values())),
        lambda: andrey.tabu(DATA, POSITIONS),
    ),
    "gfci, a mapping of DataFrames": (
        lambda: andrey.gfci(APART, confounded_index=False),
        lambda: andrey.gfci(DATA, ENV, confounded_index=False),
    ),
}


@pytest.mark.parametrize("case", list(APART_FITS))
def test_environments_apart_fit_what_their_stacked_rows_fit(case):
    apart, pooled = APART_FITS[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", andrey.ExperimentalWarning)
        assert _same(apart(), pooled())


APART_ERRORS = {
    "env with the environments apart": lambda: andrey.gies(APART, ENV, targets=TARGETS),
    "a targets matrix with the environments apart": lambda: andrey.gies(APART, targets=MASK),
    "a targets column with the environments apart": lambda: andrey.gies(
        {k: t.assign(t="") for k, t in APART.items()}, targets="t"
    ),
    "no environment": lambda: andrey.gies({}, targets={}),
    "an environment without rows": lambda: andrey.gies(
        {**APART, "set x2": DATA.iloc[:0]}, targets=TARGETS
    ),
    "DataFrames and arrays": lambda: andrey.gies(
        {**APART, "set x2": APART["set x2"].to_numpy()}, targets=TARGETS
    ),
    "DataFrames with other columns": lambda: andrey.gies(
        {**APART, "set x2": APART["set x2"].rename(columns={"x2": "x3"})}, targets=TARGETS
    ),
    "arrays with other numbers of columns": lambda: andrey.cdnod([DOMAINS[0], DOMAINS[1][:, :1]]),
    "keys that mix strings and numbers": lambda: andrey.cdnod({0: DOMAINS[0], "b": DOMAINS[1]}),
    "an environment that is not a table": lambda: andrey.cdnod({"a": DOMAINS[0], "b": [1.0, 2.0]}),
    "a list with a member that is not a table": lambda: andrey.gies(
        [APART["observed"], APART["set x1"]["x0"]], targets={0: [], 1: []}
    ),
    "a key that is not finite": lambda: andrey.cdnod({np.inf: DOMAINS[0], 1: DOMAINS[1]}),
    "a DataFrame that repeats a column name": lambda: andrey.gies(
        [pd.DataFrame(t.to_numpy(), columns=["x0", "x0", "x1"]) for t in APART.values()],
        targets={0: [], 1: [], 2: []},
    ),
    "DataFrames that repeat other column names": lambda: andrey.gies(
        [pd.DataFrame(DOMAINS[0][:, [0, 1, 0]], columns=list(c)) for c in ("aab", "abb")],
        targets={0: [], 1: []},
    ),
    "a missing value, which gies does not accept": lambda: andrey.gies(
        {**APART, "set x1": _with_nan(APART["set x1"])}, targets=TARGETS
    ),
}


@pytest.mark.parametrize("case", list(APART_ERRORS))
def test_environments_apart_that_cannot_be_read_raise(case):
    """Each refusal names the method, as a ``ValueError``, before the rows are stacked."""
    with pytest.raises(ValueError, match=r"^(gies|cdnod): "):
        APART_ERRORS[case]()


def test_a_missing_value_names_its_environment_and_row():
    nan = {**APART, "set x1": _with_nan(APART["set x1"])}
    with pytest.raises(ValueError, match=r"environment 'set x1'.* row 3,"):
        andrey.gies(nan, targets=TARGETS)


def test_cdnod_with_one_table_and_no_env_raises_as_a_missing_argument():
    with pytest.raises(TypeError):
        andrey.cdnod(X)


def test_categories_join_across_environments():
    """A hard intervention leaves its target with fewer categories in its environment; the
    stacked table keeps every column categorical, so the fit is the declared table's."""
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 600)
    b = a ^ (rng.random(600) < 0.2)
    c = b ^ (rng.random(600) < 0.2)
    table = pd.DataFrame({"a": a, "b": b, "c": c})
    table.loc[400:, "a"] = 1
    observed, knockout = (table.iloc[:400].astype("category"), table.iloc[400:].astype("category"))
    targets = {0: [], 1: ["a"]}
    apart = andrey.gies([observed, knockout], targets=targets)
    pooled = andrey.gies(table.astype("category"), np.repeat([0, 1], [400, 200]), targets=targets)
    assert _same(apart, pooled)


def test_bootstrap_resamples_environments_apart_as_their_stacked_rows():
    apart = andrey.bootstrap("gies", APART, n_resamples=2, targets=TARGETS)
    pooled = andrey.bootstrap("gies", DATA, n_resamples=2, env=ENV, targets=TARGETS)
    assert apart.structure.labels == ("x0", "x1", "x2")
    assert np.array_equal(apart.frequency("->"), pooled.frequency("->"))
    domains = andrey.bootstrap("cdnod", DOMAINS, n_resamples=2)
    labeled = np.array(["a", "b", "c"])[DOMAIN.astype(int)]  # coded 0, 1, 2, drawn within each
    assert np.array_equal(
        domains.frequency("->"),
        andrey.bootstrap("cdnod", X, n_resamples=2, env=labeled).frequency("->"),
    )


@pytest.mark.parametrize("form", ["list", "mapping"])
def test_bootstrap_draws_within_each_domain_given_apart(form):
    """Tables apart are one domain each, so cdnod's numbered positions draw within each domain,
    as text labels do, where a numeric index of one table draws its rows as a whole."""
    data = DOMAINS if form == "list" else dict(enumerate(DOMAINS))
    job = build_job(get_spec("cdnod"), data, {}, None, 0, "subsample", 0.5)
    assert job.strata is not None
    assert [len(rows) for rows in job.strata] == [len(t) for t in DOMAINS]
    assert (
        build_job(get_spec("cdnod"), X, {"env": DOMAIN}, None, 0, "subsample", 0.5).strata is None
    )


# --- the command line ---------------------------------------------------------------------------


def _run(argv, capsys):
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def files(tmp_path):
    """The chain's data with its environment as a numeric column, the environment alone, and its
    targets as JSON."""
    codes = pd.factorize(ENV)[0]
    paths = {name: tmp_path / name for name in ("data.csv", "env.csv", "t.json")}
    DATA.assign(e=codes).to_csv(paths["data.csv"], index=False)
    pd.DataFrame({"e": codes}).to_csv(paths["env.csv"], index=False)
    paths["t.json"].write_text(json.dumps({"0": [], "1": ["x1"], "2": ["x2"]}))
    DATA.to_csv(tmp_path / "plain.csv", index=False)
    paths["plain.csv"] = tmp_path / "plain.csv"
    return paths


#: The methods that read ``env``, when given, as an index no data variable causes.
OPTIONAL_ENV = ["boss", "grasp", "boss_fci", "grasp_fci", "ges", "hc", "tabu", "gfci"]


def _edges(out: str) -> list[tuple[int, int, str]]:
    structure = json.loads(out)["structure"]
    return sorted((e["source"], e["target"], e["type"]) for e in structure["edges"])


def test_cli_env_column_file_and_config_targets_fit_alike(files, tmp_path, capsys):
    config = tmp_path / "c.yaml"
    config.write_text("targets:\n  0: []\n  1: [x1]\n  2: [2]\n")
    runs = [
        ["--data", str(files["data.csv"]), "--env", "e", f"targets={files['t.json']}"],
        [
            "--data",
            str(files["plain.csv"]),
            "--env",
            str(files["env.csv"]),
            'targets={"0": [], "1": ["x1"], "2": ["x2"]}',
        ],
        ["--data", str(files["data.csv"]), "--env", "e", "--config", str(config)],
    ]
    found = []
    for argv in runs:
        code, out, _ = _run(["run", "gies", *argv, "--json"], capsys)
        assert code == 0
        assert json.loads(out)["structure"]["labels"] == ["x0", "x1", "x2"]
        found.append(_edges(out))
    assert found[0] == found[1] == found[2]
    assert found[0] == [(0, 1, "directed"), (1, 2, "directed")]


@pytest.mark.parametrize(
    ("method", "flag", "now"),
    [
        ("gies", "--interventions", "`--env` with `targets=`"),
        ("cdnod", "--domain-index", "`--env`"),
    ],
)
def test_cli_former_flags_are_refused_with_their_current_names(files, capsys, method, flag, now):
    argv = ["run", method, "--data", str(files["plain.csv"]), flag, str(files["env.csv"])]
    code, _, err = _run(argv, capsys)
    message = json.loads(err)["error"]["message"]
    assert code == 2 and f"{now} in 0.14.0" in message


@pytest.mark.parametrize("method", ["cdnod", "fci_jci"])
def test_cli_needs_env(method, files, capsys):
    """One data file cannot hold the environments apart, so the command line needs ``--env``, and
    the help schema says so."""
    code, _, err = _run(["run", method, "--data", str(files["plain.csv"]), "--json"], capsys)
    assert code == 2 and json.loads(err)["error"]["code"] == "missing_input"
    code, out, _ = _run(["run", method, "--help", "--json"], capsys)
    inputs = json.loads(out)["method"]["data"]["inputs"]
    assert code == 0 and all(i["required"] for i in inputs)


@pytest.mark.parametrize("method", ["cdnod", "fci_jci", *OPTIONAL_ENV])
def test_cli_bootstrap_takes_an_env_column(method, files, capsys):
    argv = ["bootstrap", method, "--data", str(files["data.csv"]), "--env", "e"]
    code, out, _ = _run([*argv, "--n-resamples", "2", "--json"], capsys)
    assert code == 0 and json.loads(out)["structure"]["labels"] == ["x0", "x1", "x2"]


@pytest.mark.parametrize("method", OPTIONAL_ENV)
def test_cli_env_column_and_file_fit_alike_and_name_the_index_edges(method, files, capsys):
    """``env`` is optional for the score searches and their FCI hybrids: as a column of ``--data``
    or a file it fits the same graph, and the index's edges name the variables."""
    fits = []
    for argv in (
        ["--data", str(files["data.csv"]), "--env", "e"],
        ["--data", str(files["plain.csv"]), "--env", str(files["env.csv"])],
    ):
        code, out, _ = _run(["run", method, *argv, "--json"], capsys)
        assert code == 0
        fits.append(json.loads(out))
    assert fits[0]["structure"] == fits[1]["structure"]
    assert fits[0]["metadata"]["index_edges"] == fits[1]["metadata"]["index_edges"]
    assert {variable for variable, _ in fits[0]["metadata"]["index_edges"]} <= {"x0", "x1", "x2"}
    code, out, _ = _run(["run", method, "--data", str(files["plain.csv"]), "--json"], capsys)
    assert code == 0 and "index_edges" not in json.loads(out)["metadata"]


def test_cli_refuses_ambiguous_or_doubled_inputs(files, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "e").write_text("0\n")  # a file named like the env column
    data, env = str(files["data.csv"]), str(files["env.csv"])
    usage = [
        ["run", "gies", "--data", data, "--env", "e", f"targets={files['t.json']}"],
        ["run", "gies", "--data", data, "--env", "missing.csv", f"targets={files['t.json']}"],
        ["run", "gies", "--data", data, "--env", env, "targets=[1, 2]"],
        ["run", "pc", "--data", data, "--domain-index", env],
        ["run", "fci_jci", "--data", data, "--domain-index", env],
    ]
    for argv in usage:
        code, _, err = _run(argv, capsys)
        assert code == 2, argv
        assert "error" in json.loads(err)


def test_cli_env_without_targets_is_a_data_error(files, capsys):
    argv = ["run", "gies", "--data", str(files["plain.csv"]), "--env", str(files["env.csv"])]
    code, _, err = _run(argv, capsys)
    assert code == 3 and json.loads(err)["error"]["code"] == "data_error"

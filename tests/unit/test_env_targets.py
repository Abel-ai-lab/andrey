"""The environment and target arguments of ``gies`` and ``cdnod``, in Python and on the CLI.

Every accepted form of ``env`` and ``targets`` gives the same fit as the 0/1 intervention matrix
it describes, each former form warns once and gives the same fit as its replacement, and every
form that cannot say which variables an environment intervened on raises.
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
from andrey.core import warning_policy

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


# --- gies: every form of env and targets ----------------------------------------------------------


def _positions_or_nan() -> np.ndarray:
    """Each row's target as a 0-based position, ``NaN`` on an observational row."""
    cells = np.full(N, np.nan, dtype=object)
    cells[ENV == "set x1"], cells[ENV == "set x2"] = 1, 2
    return cells


GIES_FORMS = {
    "mapping by name": lambda: andrey.gies(DATA, ENV, targets=TARGETS),
    "mapping by position": lambda: andrey.gies(
        DATA, ENV, targets={"observed": [], "set x1": [1], "set x2": 2}
    ),
    "number labels": lambda: andrey.gies(
        DATA, np.repeat([7, 3, 5], [500, 200, 200]), targets={7: [], 3: ["x1"], 5: ["x2"]}
    ),
    "env column": lambda: andrey.gies(DATA.assign(e=ENV), "e", targets=TARGETS),
    "targets column": lambda: andrey.gies(
        DATA.assign(t=np.select([ENV == "set x1", ENV == "set x2"], ["x1", "x2"], "")),
        targets="t",
    ),
    "targets column of lists": lambda: andrey.gies(
        DATA.assign(t=[[] if e == "observed" else [e[-2:]] for e in ENV]), targets="t"
    ),
    "targets column of NaN and positions": lambda: andrey.gies(
        DATA.assign(t=_positions_or_nan()), targets="t"
    ),
    "env and targets columns": lambda: andrey.gies(
        DATA.assign(e=ENV, t=[[] if e == "observed" else e[-2:] for e in ENV]), "e", targets="t"
    ),
    "matrix with env": lambda: andrey.gies(DATA, ENV, targets=MASK),
    "matrix as a table": lambda: andrey.gies(DATA, targets=pd.DataFrame(MASK.astype(int))),
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


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda: andrey.gies(DATA, interventions=MASK), id="interventions="),
        pytest.param(lambda: andrey.gies(DATA, MASK), id="the matrix as the second argument"),
    ],
)
def test_a_former_form_warns_once_and_fits_the_same(call, reference):
    warning_policy._warned.clear()
    with pytest.warns(andrey.DeprecatedArgumentWarning):
        assert _same(call(), reference)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        call()  # the message warns once


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
    "interventions and targets": lambda: andrey.gies(DATA, interventions=MASK, targets=MASK),
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


def test_bootstrap_resamples_a_targets_matrix_without_env_with_the_rows():
    matrix = andrey.bootstrap("gies", DATA, n_resamples=2, targets=MASK)
    warning_policy._warned.clear()
    with pytest.warns(andrey.DeprecatedArgumentWarning):
        former = andrey.bootstrap("gies", DATA, n_resamples=2, interventions=MASK)
    assert np.array_equal(former.frequency("->"), matrix.frequency("->"))


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


def test_cdnod_with_one_domain_is_pc():
    one = andrey.cdnod(X, np.zeros(600))
    assert np.array_equal(one.structure.to_numpy(), andrey.pc(X).structure.to_numpy())


def test_the_former_domain_index_warns_once_and_fits_the_same():
    warning_policy._warned.clear()
    with pytest.warns(andrey.DeprecatedArgumentWarning):
        assert _same(andrey.cdnod(X, domain_index=DOMAIN), andrey.cdnod(X, DOMAIN))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        andrey.cdnod(X, domain_index=DOMAIN)  # the message warns once
    with pytest.raises(ValueError):
        andrey.cdnod(X, env=DOMAIN, domain_index=DOMAIN)


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


def test_bootstrap_draws_cdnod_rows_whole_for_numbers_and_within_domains_for_text():
    """A numeric index, which cdnod reads as numbers, draws rows as a whole, as under its former
    name; text labels draw within each domain."""
    frame = pd.DataFrame(X, columns=["x", "y"])
    numbers = andrey.bootstrap("cdnod", frame, n_resamples=4, env=DOMAIN)
    warning_policy._warned.clear()
    with pytest.warns(andrey.DeprecatedArgumentWarning):
        former = andrey.bootstrap("cdnod", frame, n_resamples=4, domain_index=DOMAIN)
    assert np.array_equal(numbers.frequency("->"), former.frequency("->"))
    one_row = np.array(["a", "b", "c"])[DOMAIN.astype(int)]
    one_row[0] = "d"  # a domain of one row stays in every subsample
    assert andrey.bootstrap("cdnod", frame, n_resamples=4, env=one_row).n_resamples == 4


# --- the command line ---------------------------------------------------------------------------


def _run(argv, capsys):
    code = main(argv)
    out, err = capsys.readouterr()
    return code, out, err


@pytest.fixture
def files(tmp_path):
    """The chain's data with its environment as a numeric column, the environment alone, its
    targets as JSON, and its intervention matrix."""
    codes = pd.factorize(ENV)[0]
    paths = {name: tmp_path / name for name in ("data.csv", "env.csv", "t.json", "mask.csv")}
    DATA.assign(e=codes).to_csv(paths["data.csv"], index=False)
    pd.DataFrame({"e": codes}).to_csv(paths["env.csv"], index=False)
    paths["t.json"].write_text(json.dumps({"0": [], "1": ["x1"], "2": ["x2"]}))
    np.savetxt(paths["mask.csv"], MASK.astype(int), delimiter=",")
    DATA.to_csv(tmp_path / "plain.csv", index=False)
    paths["plain.csv"] = tmp_path / "plain.csv"
    return paths


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


def test_cli_former_flags_warn_and_fit_alike(files, capsys):
    new = ["--data", str(files["plain.csv"]), "--env", str(files["env.csv"])]
    _, out, _ = _run(["run", "gies", *new, f"targets={files['t.json']}", "--json"], capsys)
    warning_policy._warned.clear()
    former = ["--data", str(files["plain.csv"]), "--interventions", str(files["mask.csv"])]
    code, out_former, _ = _run(["run", "gies", *former, "--json"], capsys)
    assert code == 0 and _edges(out_former) == _edges(out)
    warned = json.loads(out_former)["run"]["warnings"]
    assert "DeprecatedArgumentWarning" in {w["category"] for w in warned}

    _, out, _ = _run(["run", "cdnod", *new, "--json"], capsys)
    warning_policy._warned.clear()
    former = ["--data", str(files["plain.csv"]), "--domain-index", str(files["env.csv"])]
    code, out_former, _ = _run(["run", "cdnod", *former, "--json"], capsys)
    assert code == 0 and _edges(out_former) == _edges(out)
    warned = json.loads(out_former)["run"]["warnings"]
    assert "DeprecatedArgumentWarning" in {w["category"] for w in warned}


@pytest.mark.parametrize("method", ["cdnod", "fci_jci"])
def test_cli_bootstrap_takes_an_env_column(method, files, capsys):
    argv = ["bootstrap", method, "--data", str(files["data.csv"]), "--env", "e"]
    code, out, _ = _run([*argv, "--n-resamples", "2", "--json"], capsys)
    assert code == 0 and json.loads(out)["structure"]["labels"] == ["x0", "x1", "x2"]


def test_cli_refuses_ambiguous_or_doubled_inputs(files, tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "e").write_text("0\n")  # a file named like the env column
    data, env = str(files["data.csv"]), str(files["env.csv"])
    usage = [
        ["run", "gies", "--data", data, "--env", "e", f"targets={files['t.json']}"],
        ["run", "gies", "--data", data, "--env", "missing.csv", f"targets={files['t.json']}"],
        [
            "run",
            "gies",
            "--data",
            data,
            "--interventions",
            str(files["mask.csv"]),
            f"targets={files['t.json']}",
        ],
        ["run", "cdnod", "--data", data, "--env", env, "--domain-index", env],
        ["run", "fci_jci", "--data", data, "--domain-index", env],
        ["run", "gies", "--data", data, "--env", env, "targets=[1, 2]"],
        ["run", "pc", "--data", data, "--domain-index", env],
    ]
    for argv in usage:
        code, _, err = _run(argv, capsys)
        assert code == 2, argv
        assert "error" in json.loads(err)


def test_cli_env_without_targets_is_a_data_error(files, capsys):
    argv = ["run", "gies", "--data", str(files["plain.csv"]), "--env", str(files["env.csv"])]
    code, _, err = _run(argv, capsys)
    assert code == 3 and json.loads(err)["error"]["code"] == "data_error"

"""``andrey.data.load_dataset`` downloads once, checks the SHA-256, caches, and falls back in order.

The ``offline_datasets`` fixture replaces the network: it serves Sachs's 853 observational rows
and the published ASIA network.
"""

from __future__ import annotations

import dataclasses
import hashlib
import urllib.error

import numpy as np
import pytest

from andrey.data import list_datasets, load_dataset, real
from andrey.data.sortability import index_correlation


def test_sachs_is_the_observational_rows_and_the_consensus_network(offline_datasets):
    sachs = load_dataset("sachs")
    assert sachs.data.shape == (853, 11)
    assert list(sachs.data.columns) == list(real.SACHS_NODES) == list(sachs.feature_names)
    assert sachs.graph.kind == "dag" and tuple(sachs.graph.labels) == real.SACHS_NODES
    arcs = {(s, t) for s, t, kind in sachs.graph.oriented_edges() if kind == "directed"}
    assert arcs == set(real.SACHS_ARCS) and len(sachs.graph.oriented_edges()) == 17
    assert "Science 308" in sachs.description
    assert list_datasets() == ("sachs", "asia", "alarm", "hepar2", "andes")


def test_the_array_holds_the_frames_values(offline_datasets):
    frame, array = load_dataset("sachs"), load_dataset("sachs", as_frame=False)
    assert isinstance(array.data, np.ndarray) and array.data.dtype == np.float64
    np.testing.assert_array_equal(array.data, frame.data.to_numpy())
    # The first row of the pinned source file.
    np.testing.assert_array_equal(
        array.data[0], [26.4, 13.2, 8.82, 18.3, 58.8, 6.61, 17, 414, 17, 44.9, 40]
    )


def test_a_second_load_reads_the_cache(offline_datasets, tmp_path):
    load_dataset("sachs")
    load_dataset("sachs")
    assert offline_datasets == ["https://example.invalid/sachs.txt"]
    assert (tmp_path / "cache" / "sachs" / "sachs.2005.continuous.txt").is_file()


def test_data_home_overrides_the_environment(offline_datasets, tmp_path):
    load_dataset("sachs", data_home=tmp_path / "elsewhere")
    assert (tmp_path / "elsewhere" / "sachs" / "sachs.2005.continuous.txt").is_file()
    assert not (tmp_path / "cache").exists()


def test_the_default_cache_follows_xdg(monkeypatch, tmp_path):
    monkeypatch.delenv("ANDREY_DATA_DIR", raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert real.cache_dir() == tmp_path / "andrey"
    monkeypatch.delenv("XDG_CACHE_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert real.cache_dir() == tmp_path / "home" / ".cache" / "andrey"


def test_a_changed_cached_file_is_downloaded_again(offline_datasets, tmp_path):
    target = tmp_path / "cache" / "sachs" / "sachs.2005.continuous.txt"
    target.parent.mkdir(parents=True)
    target.write_text("raf\tmek\n1\t2\n")
    assert load_dataset("sachs").data.shape == (853, 11)
    assert len(offline_datasets) == 1 and target.read_bytes() != b"raf\tmek\n1\t2\n"


def test_wrong_bytes_are_refused_with_every_url_tried(offline_datasets, monkeypatch, tmp_path):
    monkeypatch.setattr(real, "fetch", lambda url: b"not the data")
    with pytest.raises(OSError, match="SHA-256") as caught:
        load_dataset("sachs")
    assert "https://example.invalid/sachs.txt" in str(caught.value)
    assert not (tmp_path / "cache" / "sachs" / "sachs.2005.continuous.txt").exists()


def test_offline_names_the_url_and_where_to_put_the_file(offline_datasets, monkeypatch, tmp_path):
    def offline(url):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr(real, "fetch", offline)
    with pytest.raises(OSError, match="no route to host") as caught:
        load_dataset("sachs")
    message = str(caught.value)
    assert "https://example.invalid/sachs.txt" in message
    assert str(tmp_path / "cache" / "sachs" / "sachs.2005.continuous.txt") in message


def test_sources_are_tried_in_order_until_the_checksum_matches(offline_datasets, monkeypatch):
    good = real.fetch("https://example.invalid/sachs.txt")
    served = {
        "https://a.invalid/x": None,
        "https://b.invalid/x": b"wrong",
        "https://c.invalid/x": good,
    }
    tried = []

    def fetch(url):
        tried.append(url)
        if served[url] is None:
            raise urllib.error.URLError("unreachable")
        return served[url]

    remote = dataclasses.replace(
        real.DATASETS["sachs"], sources=tuple(served), sha256=hashlib.sha256(good).hexdigest()
    )
    monkeypatch.setitem(real.DATASETS, "sachs", remote)
    monkeypatch.setattr(real, "fetch", fetch)
    assert load_dataset("sachs").data.shape == (853, 11)
    assert tried == list(served)


def test_a_changed_cache_that_cannot_be_replaced_is_named(offline_datasets, monkeypatch, tmp_path):
    target = tmp_path / "cache" / "sachs" / "sachs.2005.continuous.txt"
    target.parent.mkdir(parents=True)
    target.write_text("raf\tmek\n1\t2\n")
    monkeypatch.setattr(real, "fetch", lambda url: b"not the data")
    with pytest.raises(OSError, match="left in place") as caught:
        load_dataset("sachs")
    assert f"cached file at {target}" in str(caught.value)
    assert target.read_text() == "raf\tmek\n1\t2\n"


@pytest.mark.parametrize("url", ["http://example.invalid/x", "file:///etc/passwd"])
def test_a_source_must_be_https(url):
    with pytest.raises(ValueError, match="https://"):
        real.fetch(url)


def test_an_unknown_name_lists_the_datasets():
    with pytest.raises(ValueError, match="sachs"):
        load_dataset("iris")


ASIA_ARCS = {
    ("asia", "tub"),
    ("smoke", "lung"),
    ("smoke", "bronc"),
    ("lung", "either"),
    ("tub", "either"),
    ("either", "xray"),
    ("bronc", "dysp"),
    ("either", "dysp"),
}


ASIA_NODES = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"]


def test_a_published_network_is_its_graph_with_data_simulated_on_it(offline_datasets):
    asia = load_dataset("asia")
    assert asia.data.shape == (1000, 8)
    assert sorted(asia.data.columns) == sorted(ASIA_NODES)
    assert list(asia.data.columns) == list(asia.feature_names) == list(asia.graph.labels)
    assert list(load_dataset("asia", shuffle=False).data.columns) == ASIA_NODES
    edges = asia.graph.oriented_edges()
    assert {(s, t) for s, t, _ in edges} == ASIA_ARCS and {k for *_, k in edges} == {"directed"}
    # Standardized columns, and the simulation recorded in the description.
    np.testing.assert_allclose(asia.data.std(ddof=0), 1.0)
    assert "simulated" in asia.description and "n=1000, seed=0" in asia.description
    assert "Lauritzen" in asia.description


def test_each_simulated_column_is_the_variable_it_is_named_for(offline_datasets):
    """The two ends of every arc are correlated; two variables with no path or common cause are
    not."""
    corr = load_dataset("asia").data.corr().abs()
    assert all(corr.loc[s, t] > 0.3 for s, t in ASIA_ARCS)
    unrelated = [(a, b) for a in ("asia", "tub") for b in ("smoke", "lung", "bronc")]
    assert all(corr.loc[s, t] < 0.15 for s, t in unrelated)


def test_n_and_seed_fix_the_simulated_rows(offline_datasets):
    first = load_dataset("asia", n=200, seed=3, as_frame=False).data
    np.testing.assert_array_equal(first, load_dataset("asia", n=200, seed=3, as_frame=False).data)
    assert first.shape == (200, 8)
    np.testing.assert_array_equal(
        first, load_dataset("asia", n=np.int64(200), seed=np.int64(3), as_frame=False).data
    )
    assert not np.array_equal(first, load_dataset("asia", n=200, seed=4, as_frame=False).data)
    assert "n=200, seed=4" in load_dataset("asia", n=200, seed=4).description


def test_measured_data_take_no_n_or_seed_and_nothing_is_fetched(offline_datasets):
    for options in ({"n": 100}, {"seed": 1}, {"discrete": True}):
        with pytest.raises(ValueError, match="measured"):
            load_dataset("sachs", **options)
    assert offline_datasets == []


@pytest.mark.parametrize(
    ("options", "named"),
    [
        ({"n": 1}, "n"),
        ({"n": 2.5}, "n"),
        ({"n": True}, "n"),
        ({"seed": -1}, "seed"),
        ({"seed": "0"}, "seed"),
    ],
)
def test_bad_n_or_seed_is_refused_before_any_fetch(offline_datasets, options, named):
    with pytest.raises(ValueError, match=f"^{named} must be"):
        load_dataset("asia", **options)
    assert offline_datasets == []


def test_the_bif_reader_takes_every_parent_and_checks_the_variables():
    text = """
network x { }
variable a { type discrete [ 2 ] { y, n }; }
variable b_1 { type discrete [ 2 ] { y, n }; }
variable c { type discrete [ 2 ] { y, n }; }
probability ( a ) { table 0.5, 0.5; }
probability ( b_1 ) { table 0.5, 0.5; }
probability(c|a,  b_1) { (y, y) 1.0, 0.0; }
"""
    assert real.read_bif(text) == (("a", "b_1", "c"), (("a", "c"), ("b_1", "c")))
    with pytest.raises(ValueError, match="undeclared"):
        real.read_bif(text.replace("c|a,", "c|z,"))
    with pytest.raises(ValueError, match="exactly one"):
        real.read_bif(text.replace("probability ( a ) { table 0.5, 0.5; }", ""))


def test_a_published_network_is_sampled_from_its_own_tables(offline_datasets):
    """ASIA's tables fix each variable's chance of "yes": 0.5 for smoke, 0.055 for lung (0.1 for a
    smoker, 0.01 otherwise), 0.45 for bronc. 20,000 rows put each within 0.015 of it."""
    asia = load_dataset("asia", n=20_000, seed=0, discrete=True)
    assert asia.data.shape == (20_000, 8)
    assert all(asia.data[c].cat.categories.tolist() == ["yes", "no"] for c in asia.data.columns)
    share = {c: float((asia.data[c] == "yes").mean()) for c in ("smoke", "lung", "bronc")}
    assert share == pytest.approx({"smoke": 0.5, "lung": 0.055, "bronc": 0.45}, abs=0.015)
    # either is lung OR tub in every row: its table is deterministic.
    either = (asia.data["lung"] == "yes") | (asia.data["tub"] == "yes")
    assert ((asia.data["either"] == "yes") == either).all()
    assert "own conditional probability tables (n=20000, seed=0)" in asia.description


def test_the_array_holds_the_category_codes_and_the_seed_fixes_them(offline_datasets):
    frame = load_dataset("asia", n=300, seed=2, discrete=True).data
    array = load_dataset("asia", n=300, seed=2, discrete=True, as_frame=False).data
    assert array.dtype == np.float64
    np.testing.assert_array_equal(array, np.column_stack([frame[c].cat.codes for c in frame]))
    again = load_dataset("asia", n=300, seed=2, discrete=True, as_frame=False).data
    np.testing.assert_array_equal(array, again)
    other = load_dataset("asia", n=300, seed=3, discrete=True, as_frame=False).data
    assert not np.array_equal(array, other)


_TABLES = """
network x { }
variable a { type discrete [ 2 ] { lo, hi }; }
variable b { type discrete [ 3 ] { r, g, b }; }
variable c { type discrete [ 2 ] { no, yes }; }
probability ( a ) { table 0.4, 0.6; }
probability ( c | b, a ) {
  (r, lo) 1.0, 0.0; (r, hi) 0.0, 1.0;
  (g, lo) 1.0, 0.0; (g, hi) 0.0, 1.0;
  (b, lo) 0.0, 1.0; (b, hi) 1.0, 0.0;
}
probability ( b ) { table 0.2, 0.3, 0.5; }
"""


def test_the_table_reader_numbers_rows_by_the_parents_in_their_listed_order():
    tables = real.read_bif_tables(_TABLES)
    assert tables.nodes == ("a", "b", "c")
    assert tables.states == (("lo", "hi"), ("r", "g", "b"), ("no", "yes"))
    assert tables.parents == ((), (), (1, 0))
    # Row (b, a) = 2 * b + a: the first listed parent is the most significant digit.
    np.testing.assert_array_equal(tables.tables[2][:, 1], [0, 1, 0, 1, 1, 0])
    codes = tables.sample(2000, np.random.default_rng(0)).astype(int)
    a, b, c = codes.T
    np.testing.assert_array_equal(c, np.where(b == 2, 1 - a, a))  # c's deterministic table
    assert set(np.unique(b)) == {0, 1, 2}


@pytest.mark.parametrize(
    ("change", "named"),
    [
        (("(b, lo) 0.0, 1.0;", ""), "misses a parent configuration"),
        (("(b, lo) 0.0, 1.0;", "(b, lo) 0.5, 0.6;"), "must be a distribution"),
        (("(b, lo) 0.0, 1.0;", "(b, mid) 0.0, 1.0;"), "category 'mid' of 'a'"),
        (("(b, lo) 0.0, 1.0;", "(b, lo) 1.0;"), "malformed"),
        (("(r, lo) 1.0, 0.0;", "table 1.0, 0.0;"), "malformed"),
        (("type discrete [ 3 ] { r, g, b };", "type continuous;"), "not discrete"),
        (("probability ( b ) { table 0.2, 0.3, 0.5; }", ""), "no probability block"),
        (("c | b, a", "c | b, z"), "undeclared"),
    ],
)
def test_the_table_reader_refuses_an_incomplete_table(change, named):
    with pytest.raises(ValueError, match=named):
        real.read_bif_tables(_TABLES.replace(*change))


@pytest.mark.parametrize("discrete", [False, True])
def test_the_shuffle_relabels_a_published_network(offline_datasets, discrete):
    """The shuffled rows are the unshuffled ones under the same names, with the same graph."""
    shuffled = load_dataset("asia", n=300, seed=5, discrete=discrete)
    plain = load_dataset("asia", n=300, seed=5, discrete=discrete, shuffle=False)
    assert list(shuffled.feature_names) != ASIA_NODES
    assert shuffled.data[list(plain.feature_names)].equals(plain.data)
    assert set(shuffled.graph.oriented_edges()) == set(plain.graph.oriented_edges())
    assert "order drawn from the seed" in shuffled.description
    assert "keep the file's variable order" in plain.description


def test_measured_data_keep_the_file_column_order(offline_datasets):
    sachs = load_dataset("sachs")
    assert sachs.feature_names == real.SACHS_NODES
    assert load_dataset("sachs", shuffle=False).data.equals(sachs.data)


def _asia_correlation(seed: int, shuffle: bool) -> float:
    asia = load_dataset("asia", n=2, seed=seed, as_frame=False, shuffle=shuffle)
    column = {name: j for j, name in enumerate(asia.feature_names)}
    edges = np.array([(column[a], column[b]) for a, b, _ in asia.graph.oriented_edges()])
    return index_correlation(edges, len(column))


def test_a_published_network_column_order_says_nothing_of_the_causal_order(offline_datasets):
    """ASIA's file lists its variables in causal order (a correlation of 0.90 with their depth).

    Under a uniform column order the rank correlation of column index and causal depth has mean 0
    and variance ``1 / 7`` on ASIA's eight columns, so its mean over 200 seeds stays within four
    standard deviations, ``4 / sqrt(200 * 7)``, of 0. The file's order, the positive control, does
    not.
    """
    seeds = range(200)
    bound = 4 / np.sqrt(len(seeds) * 7)
    assert abs(np.mean([_asia_correlation(s, True) for s in seeds])) < bound
    assert _asia_correlation(0, False) > bound

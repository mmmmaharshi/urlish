"""Offline checks: no data files, no network."""

from __future__ import annotations

import contextlib
import io
import re
import tempfile
from pathlib import Path

import numpy as np
import pytest

from urlish.core import (
    canonicalize,
    fold,
    host_of,
    levenshtein,
    registrable_domain,
    split_url,
)
from urlish.data import (
    Row,
    group_split,
    load_majestic,
    load_tranco,
    load_url_lines,
    random_split,
    remove_domains,
    select,
)
from urlish.features import FEATURE_NAMES, BrandIndex, analyze, featurize
from urlish.metrics import at_threshold, threshold_at_fpr

BRANDS = BrandIndex(
    [
        ("google", "google.com"),
        ("paypal", "paypal.com"),
        ("microsoft", "microsoft.com"),
        ("amazon", "amazon.com"),
        ("sbi", "onlinesbi.sbi"),
        ("instagram", "instagram.com"),
        ("tubi", "tubi.io"),
        ("adgrx", "adgrx.com"),
    ]
)


def test_domains() -> None:
    assert registrable_domain("login.secure.example.co.uk") == "example.co.uk"
    assert registrable_domain("a.b.c.com") == "c.com"
    assert registrable_domain("EXAMPLE.COM.") == "example.com"
    assert registrable_domain("10.0.0.1") == "10.0.0.1"
    assert registrable_domain("localhost") == "localhost"


def test_folding() -> None:
    assert fold("p\u0430ypa1") == "paypal", fold("p\u0430ypa1")  # Cyrillic a
    assert fold("g00gle") == "google"
    assert fold("Pay-Pal.com") == "paypalcom"
    assert levenshtein("paypal", "paypal") == 0
    assert levenshtein("paypa1", "paypal", 2) <= 2
    assert levenshtein("google", "amazon", 2) == 3  # capped


def test_impersonation() -> None:
    def brands_of(url: str) -> list[str]:
        return [m.brand for m in analyze(url, BRANDS)[0]]

    assert brands_of("http://paypa1.com/login") == ["paypal"]
    assert brands_of("http://www.paypal.com/") == [], "the real brand must not fire"
    assert brands_of("http://paypal.com.secure-login.xyz/") == ["paypal"]
    assert brands_of("http://google.xyz/") == ["google"], "same label, wrong TLD"
    assert brands_of("http://gooogle.com/") == ["google"]
    assert brands_of("http://example.org/") == []

    # Short labels need an exact match: two edits is noise at that length.
    assert brands_of("http://www.sbi.co.in/") == [], "3-char label must not match"
    assert brands_of("http://a-9krx.shop/") == [], "5-char label, two edits, no match"

    matches, dist = analyze("http://paypa1.com/", BRANDS)
    assert dist == 0, dist  # digit-for-letter fold makes it an exact skeleton match
    assert matches[0].kind == "confusable", matches[0].kind

    # A Cyrillic confusable takes the same path as a digit swap.
    assert analyze("http://p\u0430ypal.com/", BRANDS)[0][0].kind == "confusable"

    _, typo = analyze("http://gooogle.com/", BRANDS)
    assert typo == 1, typo


def test_parse() -> None:
    """One parse path: malformed input must not split one way for features and
    another way for hosts, which is what the duplicated parsers did."""
    assert split_url("http://[::1").netloc == "", "broken bracket must split empty"
    assert host_of("http://[::1") == ""

    for url in ("", "not a url", "http://[::1]/", "ftp://x"):
        assert host_of(url) == (split_url(url).hostname or "").strip().lower().rstrip("."), url
        featurize(url, BRANDS)

    assert canonicalize("HTTP://EXAMPLE.COM./Path/") == "example.com/Path"


def test_features() -> None:
    vec = featurize("https://paypa1.com/verify/account?token=1", BRANDS)
    assert len(vec) == len(FEATURE_NAMES), (len(vec), len(FEATURE_NAMES))
    assert all(isinstance(v, float) for v in vec)
    # Garbage in, finite numbers out.
    for bad in ("", "not a url", "http://[::1]/", "ftp://x", "\u0430" * 300):
        v = featurize(bad, BRANDS)
        assert len(v) == len(FEATURE_NAMES)
        assert all(np.isfinite(v)), bad

    # Scheme and trailing slash must not change any feature. If they do, the
    # model learns how the training set was assembled.
    forms = [
        "http://a-9krx.shop",
        "https://a-9krx.shop/",
        "a-9krx.shop",
        "HTTPS://A-9KRX.SHOP/",
    ]
    vecs = {tuple(featurize(f, BRANDS)) for f in forms}
    assert len(vecs) == 1, f"URL form changed the features: {vecs}"


def test_threshold() -> None:
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.random(900) * 0.85, rng.random(100) * 0.1 + 0.9])
    y = np.array([0] * 900 + [1] * 100)
    thr = threshold_at_fpr(scores, y, 0.01)
    m = at_threshold(scores, y, thr)
    assert m["fpr"] <= 0.01, m["fpr"]
    assert m["recall"] > 0.9, m["recall"]

    # A perfect separation must not be capped below the target.
    assert threshold_at_fpr(np.array([0.1, 0.2, 0.9]), np.array([0, 0, 1]), 0.01) > 0.2


def test_provenance() -> None:
    """A row's collection and stratum are separate facts, and `select` keeps them
    apart. Conflating them is what makes a collection split measure popularity."""
    rows = [
        Row("http://a.com/", 1, "a.com", "phishing_army", None),
        Row("http://b.com/", 1, "b.com", "openphish", None),
        Row("http://c.com/", 0, "c.com", "tranco", "head"),
        Row("http://d.com/", 0, "d.com", "tranco", "tail"),
        Row("http://e.com/", 0, "e.com", "majestic", "head"),
    ]
    assert select(rows, collections={"tranco"}) == [2, 3]
    assert select(rows, stratum="head") == [2, 4]
    assert select(rows, label=0, collections={"majestic"}) == [4]
    assert select(rows) == [0, 1, 2, 3, 4]

    # Domains shared across collections must be removable, or a collection split
    # is measuring leakage: Majestic and Tranco overlap heavily in the head band.
    assert remove_domains([2, 3, 4], rows, {"c.com"}) == [3, 4]


def test_loaders() -> None:
    """A bad collection must stop the run, not produce a plausible-looking report."""
    with tempfile.TemporaryDirectory() as tmp:
        empty = Path(tmp) / "empty.csv"
        empty.write_text("", encoding="utf-8")
        for loader in (load_tranco, load_majestic, load_url_lines):
            with pytest.raises(ValueError, match=re.escape(str(empty))):
                loader(empty)

        header_only = Path(tmp) / "header.csv"
        header_only.write_text("GlobalRank,Tld,Domain\n", encoding="utf-8")
        with pytest.raises(ValueError, match="GlobalRank,Domain"):
            load_majestic(header_only)

        good = Path(tmp) / "tranco.csv"
        good.write_text("1,example.com\nnope,example.org\n", encoding="utf-8")
        with contextlib.redirect_stderr(io.StringIO()) as err:
            assert load_tranco(good) == [(1, "example.com")]
        assert "skipped 1" in err.getvalue(), err.getvalue()


def test_splits() -> None:
    rows = [
        Row(f"http://{d}/", int(i % 2), d, "synthetic", None)
        for i, d in enumerate(f"site{n}.com" for n in range(200))
    ]
    train, test = group_split(rows, 0.2, 0)
    seen = {rows[i].domain for i in train} & {rows[i].domain for i in test}
    assert not seen, f"domain leakage: {list(seen)[:3]}"
    assert len(test) > 0 and len(train) > 0

    train_r, test_r = random_split(rows, 0.2, 0)
    assert len(train_r) + len(test_r) == len(rows)
    assert set(train_r).isdisjoint(test_r)

"""Feature extraction and brand-lookalike detection.

Two independent signals live here:

* `featurize()` produces the numeric vector a classifier learns from.
* `BrandIndex` + `explain()` implement deterministic impersonation detection.
  That part needs no training data and no model, which is the whole point:
  a lexical classifier cannot see that `paypal-secure.xyz` is *about PayPal*.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import SplitResult

from .core import (
    canonicalize,
    deletes,
    entropy,
    fold,
    has_ip_host,
    host_of,
    levenshtein,
    registrable_domain,
    split_url,
    tokenize,
)

LURE_WORDS = frozenset(
    """
    login signin sign-in verify verification secure account update confirm
    banking bank wallet invoice support billing password recover unlock
    gift bonus claim alert suspend limited validate authentication
    """.split()
)

# Longest match wins: "banking" already contains "bank", and counting both made
# one lure word look like two.
LURE_CORE = frozenset(
    word for word in LURE_WORDS if not any(word != other and word in other for other in LURE_WORDS)
)

# Cheap-registration TLDs that are over-represented in abuse feeds. This is a
# feature, not a rule: plenty of legitimate sites use them.
SUSPICIOUS_TLDS = frozenset(
    """
    zip mov top xyz click link shop live icu buzz rest cam surf monster
    quest cfd sbs lol boats autos beauty hair makeup skin bond cyou
    """.split()
)

# Labels that are too generic to treat as brands, or they would fire on
# every other domain.
GENERIC_LABELS = frozenset(
    """
    mail docs drive cloud app apps login secure support account update
    www web smtp news blog shop store video music games play
    """.split()
)


def allowed_distance(label: str) -> int:
    """Shorter labels need exact-or-one-edit matches.

    Without this, three and four character names collide with unrelated brands:
    `sbi.co.in` is two edits from `tubi`, which is not an impersonation.
    """
    return 1 if len(label) < 8 else 2


@dataclass(frozen=True)
class BrandMatch:
    brand: str
    primary: str
    kind: str  # "typosquat" | "confusable" | "brand-token"
    distance: int


class BrandIndex:
    """Brand lookalike index backed by a delete-2 candidate table.

    `entries` is an iterable of (label, primary_domain). The primary domain is
    the highest-ranked real domain for that label, used to avoid flagging the
    brand itself.
    """

    def __init__(self, entries: Iterable[tuple[str, str]], max_dist: int = 2) -> None:
        self.max_dist = max_dist
        self.primary: dict[str, str] = {}
        self.labels: frozenset[str] = frozenset()
        table: dict[str, list[str]] = {}
        for label, domain in entries:
            label = fold(label)
            if len(label) < 4 or label in GENERIC_LABELS:
                continue
            self.primary.setdefault(label, domain)
            for key in deletes(label, max_dist):
                table.setdefault(key, []).append(label)
        self.labels = frozenset(self.primary)
        self._table = {k: tuple(v) for k, v in table.items()}

    @classmethod
    def from_tranco(cls, path: str | Path, top: int = 2000, max_dist: int = 2) -> BrandIndex:
        entries: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                parts = line.strip().split(",")
                if len(parts) < 2:
                    continue
                try:
                    rank = int(parts[0])
                except ValueError:
                    continue
                if rank > top:
                    break
                reg = registrable_domain(parts[1])
                label = reg.split(".")[0] if reg else ""
                if label:
                    entries.append((label, reg))
        return cls(entries, max_dist)

    def nearest(self, candidate: str) -> tuple[int, str | None]:
        best, best_label = self.max_dist + 1, None
        for key in deletes(candidate, self.max_dist):
            for label in self._table.get(key, ()):
                dist = levenshtein(candidate, label, self.max_dist)
                if dist < best:
                    best, best_label = dist, label
                    if best == 0:
                        return best, best_label
        return best, best_label

    def __len__(self) -> int:
        return len(self.primary)

    def __contains__(self, label: str) -> bool:
        return fold(label) in self.labels


def _tld(reg: str) -> str:
    return reg.rsplit(".", 1)[-1] if "." in reg else ""


def _has_port(parts: SplitResult) -> bool:
    """A bracketed IPv6 host has colons but no port."""
    try:
        return parts.port is not None
    except ValueError:  # a non-numeric port is not a port
        return False


def explain(url: str, brands: BrandIndex | None) -> list[BrandMatch]:
    """Deterministic impersonation signals. No model, no training data."""
    return analyze(url, brands)[0]


def analyze(url: str, brands: BrandIndex | None) -> tuple[list[BrandMatch], int]:
    """Impersonation matches plus the raw distance to the nearest brand.

    Single pass so feature extraction does not repeat the lookalike search.
    """
    if brands is None or not url:
        return [], 9
    parts = split_url(url)
    host = host_of(url)
    reg = registrable_domain(host)
    if not reg:
        return [], 9
    label = reg.split(".")[0]
    folded = fold(label)
    matches: list[BrandMatch] = []

    # 1. The registrable label itself is a near-miss of a known brand.
    dist, brand = brands.nearest(folded)
    if brand and len(folded) >= 4 and dist <= allowed_distance(folded):
        primary = brands.primary[brand]
        if not (dist == 0 and reg == primary):
            kind = "confusable" if folded == brand and label != folded else "typosquat"
            matches.append(BrandMatch(brand, primary, kind, dist))

    # 2. A brand name appears as a token in a subdomain or path, but the
    #    registrable domain is not the brand's own domain.
    haystack = ".".join(host.split(".")[:-2]) if host.count(".") >= 2 else ""
    haystack = f"{haystack} {parts.path} {parts.query}"
    for tok in tokenize(haystack):
        folded_tok = fold(tok)
        if folded_tok in brands.labels and brands.primary[folded_tok] != reg:
            matches.append(BrandMatch(folded_tok, brands.primary[folded_tok], "brand-token", 0))
            break

    return matches, dist


FEATURE_NAMES = (
    "url_len",
    "host_len",
    "path_len",
    "host_entropy",
    "url_entropy",
    "digit_ratio",
    "host_hyphens",
    "host_dots",
    "subdomains",
    "path_depth",
    "query_params",
    "has_ip",
    "is_punycode",
    "has_at",
    "has_port",
    "tld_suspicious",
    "tld_len",
    "lure_count",
    "lure_in_path",
    "digits_in_host",
    "longest_token",
    "brand_distance",
    "brand_token",
    "brand_and_lure",
    "hyphen_in_registrable",
)


def featurize(url: str, brands: BrandIndex | None = None) -> list[float]:
    """Fixed-order numeric vector. Names match FEATURE_NAMES."""
    parts = split_url(url)
    host = host_of(url)
    reg = registrable_domain(host)
    label = reg.split(".")[0] if reg else ""
    canonical = canonicalize(url)
    tail = canonical[len(host) :] if host and canonical.startswith(host) else ""
    path, _, query = tail.partition("?")
    full = canonical.lower()

    lure_hits = sum(1 for w in LURE_CORE if w in full)
    longest = max((len(t) for t in host.replace(".", " ").split()), default=0)

    matches, dist = analyze(url, brands)
    brand_token = 1.0 if any(m.kind == "brand-token" for m in matches) else 0.0
    brand_lure = 1.0 if (matches and lure_hits) else 0.0

    return [
        float(len(full)),
        float(len(host)),
        float(len(path)),
        entropy(host),
        entropy(full),
        (sum(c.isdigit() for c in full) / len(full)) if full else 0.0,
        float(host.count("-")),
        float(host.count(".")),
        float(max(0, host.count(".") - 1)),
        float(path.strip("/").count("/") + 1) if path.strip("/") else 0.0,
        float(query.count("=")),
        float(has_ip_host(host)),
        float("xn--" in host),
        float("@" in url),
        float(_has_port(parts)),
        float(_tld(reg) in SUSPICIOUS_TLDS),
        float(len(_tld(reg))),
        float(lure_hits),
        float(any(w in path for w in LURE_WORDS)),
        float(sum(c.isdigit() for c in host)),
        float(longest),
        min(dist, 3) / 3.0,
        brand_token,
        brand_lure,
        float("-" in label),
    ]

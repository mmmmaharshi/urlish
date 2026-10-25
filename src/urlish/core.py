"""Pure string utilities: URL parsing, registrable domain, homoglyph folding.

No network calls, no ML. This module is deliberately dependency-free so the
same code can run in a browser (via Pyodide) as on a server.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import SplitResult, urlsplit

# shortcut: a ~70-entry Public Suffix List subset covering the common ccTLD
# second-level suffixes. Only affects `registrable_domain`. Swap in a bundled
# PSL snapshot if you need full coverage.
MULTI_SUFFIXES = frozenset(
    """
    co.uk org.uk ac.uk gov.uk me.uk ltd.uk plc.uk net.uk sch.uk
    co.in net.in org.in firm.in gen.in ind.in ac.in edu.in gov.in res.in
    com.au net.au org.au edu.au gov.au id.au asn.au
    co.nz net.nz org.nz govt.nz ac.nz
    co.jp ne.jp or.jp ac.jp go.jp ad.jp ed.jp gr.jp lg.jp
    com.br net.br org.br gov.br edu.br art.br blog.br
    com.mx net.mx org.mx gob.mx edu.mx com.ar net.ar org.ar gob.ar edu.ar
    co.za net.za org.za gov.za ac.za web.za
    com.sg net.sg org.sg gov.sg edu.sg per.sg
    com.hk net.hk org.hk gov.hk edu.hk idv.hk
    com.tw net.tw org.tw gov.tw edu.tw idv.tw
    co.kr ne.kr or.kr re.kr go.kr ac.kr
    com.tr net.tr org.tr gov.tr edu.tr av.tr bel.tr
    com.cn net.cn org.cn gov.cn edu.cn ac.cn
    com.my net.my org.my gov.my edu.my
    co.id net.id org.id web.id ac.id sch.id go.id
    com.ph net.ph org.ph gov.ph edu.ph
    com.pk net.pk org.pk gov.pk edu.pk
    com.bd net.bd org.bd gov.bd edu.bd
    com.ng net.ng org.ng gov.ng edu.ng
    co.ke or.ke ne.ke go.ke ac.ke sc.ke
    com.eg net.eg org.eg gov.eg edu.eg
    com.sa net.sa org.sa gov.sa edu.sa
    co.il net.il org.il gov.il ac.il muni.il
    com.ua net.ua org.ua gov.ua edu.ua in.ua
    com.ru net.ru org.ru msk.ru spb.ru
    co.th net.th or.th in.th go.th ac.th
    com.vn net.vn org.vn gov.vn edu.vn
    com.co net.co org.co gov.co edu.co
    com.pe net.pe org.pe gob.pe edu.pe
    com.ve net.ve org.ve gob.ve edu.ve
    com.ec net.ec org.ec gob.ec edu.ec
    com.uy net.uy org.uy gub.uy edu.uy
    com.do net.do org.do gob.do edu.do
    com.gt net.gt org.gt gob.gt edu.gt
    com.py net.py org.py gov.py edu.py
    com.bo net.bo org.bo gob.bo edu.bo
    com.cy net.cy org.cy gov.cy ac.cy
    com.mt net.mt org.mt gov.mt edu.mt
    com.gr net.gr org.gr gov.gr edu.gr
    com.pt net.pt org.pt gov.pt edu.pt
    com.es net.es org.es gob.es edu.es
    com.pl net.pl org.pl gov.pl edu.pl
    com.ro net.ro org.ro gov.ro edu.ro
    com.hr net.hr org.hr gov.hr edu.hr
    com.ba net.ba org.ba gov.ba edu.ba
    com.mk net.mk org.mk gov.mk edu.mk
    com.al net.al org.al gov.al edu.al
    com.ge net.ge org.ge gov.ge edu.ge
    com.am net.am org.am gov.am edu.am
    com.az net.az org.az gov.az edu.az
    com.kz net.kz org.kz gov.kz edu.kz
    com.by net.by org.by gov.by edu.by
    com.md net.md org.md gov.md edu.md
    """.split()
)

_IPV4 = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")
_SPLIT_TOKENS = re.compile(r"[^a-z0-9]+")

# Homoglyph folding table. Covers Cyrillic/Greek letters that render like
# Latin, plus the digit-for-letter swaps common in lookalike domains.
_HOMO = {
    "\u0430": "a",
    "\u0435": "e",
    "\u043e": "o",
    "\u0440": "p",
    "\u0441": "c",
    "\u0443": "y",
    "\u0445": "x",
    "\u0456": "i",
    "\u0458": "j",
    "\u0455": "s",
    "\u04bb": "h",
    "\u0501": "d",
    "\u04cf": "l",
    "\u03bd": "v",
    "\u03bf": "o",
    "\u03b1": "a",
    "\u03b9": "i",
    "\u03ba": "k",
    "\u03c1": "p",
    "\u03c4": "t",
    "\u03c5": "u",
    "\u03c7": "x",
    "\u03b5": "e",
    "0": "o",
    "1": "l",
    "3": "e",
    "4": "a",
    "5": "s",
    "7": "t",
    "8": "b",
    "9": "g",
    "@": "a",
    "$": "s",
    "!": "i",
    "|": "l",
}


def split_url(url: str) -> SplitResult:
    """Split a URL, adding a scheme when missing. Never raises.

    Malformed input splits to an empty result so every caller degrades the same
    way instead of each inventing its own fallback.
    """
    raw = (url or "").strip()
    if not raw:
        return urlsplit("")
    if "://" not in raw:
        raw = "http://" + raw
    try:
        return urlsplit(raw)
    except ValueError:
        return urlsplit("")


def host_of(url: str) -> str:
    """Lowercase hostname of a URL, tolerating scheme-less input."""
    if not (url or "").strip():
        return ""
    try:
        return (split_url(url).hostname or "").strip().lower().rstrip(".")
    except ValueError:
        return ""


def canonicalize(url: str) -> str:
    """Scheme-free, slash-normalised form used for length and entropy features.

    The scheme carries no signal, and including it teaches the model how the
    dataset was assembled instead of what a URL looks like.
    """
    host = host_of(url)
    if not host:
        return (url or "").strip().lower()
    parts = split_url(url)
    path = (parts.path or "").rstrip("/")
    query = f"?{parts.query}" if parts.query else ""
    return f"{host}{path}{query}"


def unicode_host(host: str) -> str:
    """Decode punycode labels so homoglyphs become visible to `fold`."""
    if "xn--" not in host:
        return host
    try:
        return host.encode("ascii").decode("idna")
    except (UnicodeError, ValueError):
        return host


def registrable_domain(host: str) -> str:
    """eTLD+1, e.g. 'login.secure.co.uk' -> 'secure.co.uk'."""
    host = unicode_host((host or "").strip().lower().rstrip("."))
    if not host or _IPV4.match(host) or ":" in host:
        return host
    labels = host.split(".")
    if len(labels) < 2:
        return host
    if len(labels) >= 3 and ".".join(labels[-2:]) in MULTI_SUFFIXES:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def has_ip_host(host: str) -> bool:
    return bool(_IPV4.match(host)) or (":" in host and host.count(":") > 1)


def entropy(text: str) -> float:
    """Shannon entropy in bits per character."""
    if not text:
        return 0.0
    n = len(text)
    return -sum((c / n) * math.log2(c / n) for c in Counter(text).values())


def fold(text: str) -> str:
    """Lowercase, map homoglyphs, drop everything that is not alphanumeric.

    'p\u0430ypa1' -> 'paypal'; 'g00gle' -> 'google'.
    """
    out = []
    for ch in (text or "").lower():
        ch = _HOMO.get(ch, ch)
        if ch.isalnum():
            out.append(ch)
    return "".join(out)


def tokenize(text: str) -> list[str]:
    return [t for t in _SPLIT_TOKENS.split((text or "").lower()) if t]


def deletes(word: str, max_dist: int) -> set[str]:
    """All strings reachable by <= max_dist single-character deletions.

    Used as a cheap candidate index so lookalike search stays linear in
    practice instead of comparing every brand against every URL.
    """
    out = {word}
    frontier = {word}
    for _ in range(max_dist):
        nxt: set[str] = set()
        for w in frontier:
            for i in range(len(w)):
                nxt.add(w[:i] + w[i + 1 :])
        out |= nxt
        frontier = nxt
    return out


def levenshtein(a: str, b: str, max_dist: int = 2) -> int:
    """Banded edit distance. Returns max_dist + 1 when the true distance is
    larger, which is all the caller needs for thresholding."""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > max_dist:
        return max_dist + 1
    inf = max_dist + 1
    prev = [min(j, inf) for j in range(lb + 1)]
    for i in range(1, la + 1):
        cur = [inf] * (lb + 1)
        cur[0] = min(i, inf)
        lo = max(1, i - max_dist)
        hi = min(lb, i + max_dist)
        for j in range(lo, hi + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            best = prev[j] + 1
            if cur[j - 1] + 1 < best:
                best = cur[j - 1] + 1
            if prev[j - 1] + cost < best:
                best = prev[j - 1] + cost
            cur[j] = best if best < inf else inf
        prev = cur
    return prev[lb]

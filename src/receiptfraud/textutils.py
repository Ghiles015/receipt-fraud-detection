"""Normalisation de texte, détection de montants et de dates."""
from __future__ import annotations

import re
from datetime import date
from difflib import SequenceMatcher

MONTHS = {m: i + 1 for i, m in enumerate(
    ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"])}

# Montant : 2 décimales obligatoires (format des reçus), séparateur de milliers optionnel.
AMOUNT_RE = re.compile(r"(?<![\d.])-?(?:\d{1,3}(?:,\d{3})+|\d+)[.,]\d{2}(?![\d%])")

_D = r"(\d{1,2})"
_Y = r"(\d{4}|\d{2})"
_MON = r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*"
DATE_PATTERNS = [
    # (regex, ordre des groupes)
    (re.compile(rf"(?<!\d)(\d{{4}})\s*[-/.]\s*(\d{{1,2}})\s*[-/.]\s*(\d{{1,2}})(?!\d)"), "ymd"),
    (re.compile(rf"(?<!\d){_D}\s*[-/.]\s*(\d{{1,2}})\s*[-/.]\s*{_Y}(?!\d)"), "dmy"),
    (re.compile(rf"(?<!\d){_D}\s*[-/ ]?\s*{_MON}\s*[-/,' ]?\s*{_Y}(?!\d)", re.I), "dMy"),
    (re.compile(rf"{_MON}\s*{_D},?\s*{_Y}(?!\d)", re.I), "Mdy"),
]


def norm(s: str) -> str:
    """Majuscules + espaces fusionnés (normalisation utilisée pour l'évaluation)."""
    return re.sub(r"\s+", " ", (s or "").upper()).strip()


def norm_nospace(s: str) -> str:
    return re.sub(r"\s+", "", (s or "").upper())


def tokens(s: str) -> list[str]:
    return re.findall(r"[A-Z0-9]+", norm(s))


def similarity(a: str, b: str) -> float:
    a, b = norm(a), norm(b)
    if not a and not b:
        return 1.0
    return SequenceMatcher(None, a, b).ratio()


def parse_amount(tok: str) -> float | None:
    tok = tok.strip()
    m = re.fullmatch(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)[.,]\d{2}", tok)
    if not m:
        return None
    body, dec = tok[:-3], tok[-2:]
    body = body.replace(",", "")
    try:
        return float(f"{body}.{dec}")
    except ValueError:
        return None


def find_amounts(s: str) -> list[tuple[str, float]]:
    out = []
    for m in AMOUNT_RE.finditer(s.replace(" .", ".").replace(". ", ".")):
        v = parse_amount(m.group(0))
        if v is not None:
            out.append((m.group(0), v))
    return out


def label_amount(s: str) -> float | None:
    """Montant d'un champ total de vérité terrain (« RM9.00 », « 9.00 », « 1,234.50 »)."""
    am = find_amounts(s)
    if am:
        return am[-1][1]
    m = re.search(r"\d+(?:\.\d+)?", s.replace(",", ""))
    return float(m.group(0)) if m else None


def _year(y: str) -> int:
    y = int(y)
    return y + 2000 if y < 100 else y


def find_dates(s: str) -> list[tuple[str, tuple[int, int, int] | None]]:
    """Retourne les chaînes de date trouvées et leur (année, mois, jour) si interprétable.
    Une date « impossible » (31/02) est renvoyée avec None : utile pour la détection de fraude."""
    found = []
    up = s.upper()
    taken: list[tuple[int, int]] = []
    for rx, order in DATE_PATTERNS:
        for m in rx.finditer(up):
            if any(a < m.end() and m.start() < b for a, b in taken):
                continue
            g = m.groups()
            try:
                if order == "ymd":
                    y, mo, d = int(g[0]), int(g[1]), int(g[2])
                elif order == "dmy":
                    d, mo, y = int(g[0]), int(g[1]), _year(g[2])
                elif order == "dMy":
                    d, mo, y = int(g[0]), MONTHS[g[1][:3].upper()], _year(g[2])
                else:
                    mo, d, y = MONTHS[g[0][:3].upper()], int(g[1]), _year(g[2])
            except (KeyError, ValueError):
                continue
            taken.append((m.start(), m.end()))
            ymd = (y, mo, d)
            try:
                date(y, mo, d)
            except ValueError:
                # Format numérique ambigu : tester mm/dd/yyyy avant de déclarer la date invalide.
                if order == "dmy":
                    try:
                        date(y, d, mo)
                        ymd = (y, d, mo)
                    except ValueError:
                        ymd = None
                else:
                    ymd = None
            found.append((s[m.start():m.end()].strip(), ymd))
    return found


_REG_RE = re.compile(
    r"\s*[\(\[]?\s*(?:CO\.?\s*(?:REG|NO)[^)]*|\d{4,8}\s*-?\s*[A-Z]|[A-Z]{2}\d{5,}-?[A-Z]?)\s*[\)\]]?\s*$", re.I)


def clean_company(s: str) -> str:
    """Retire le numéro d'immatriculation collé au nom (« 99 SPEED MART S/B (519537-X) »)."""
    s = norm(s)
    prev = None
    while prev != s:
        prev = s
        s = _REG_RE.sub("", s).strip(" :-")
    return s

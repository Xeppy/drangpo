"""Terminal rendering. Shows the answer with provenance colouring and the
certificate underneath, so the guarantee is visible, not buried in a log."""
from __future__ import annotations
import sys

from .types import Answer

_C = {"grey": "\033[90m", "red": "\033[31m", "green": "\033[32m",
      "yellow": "\033[33m", "cyan": "\033[36m", "bold": "\033[1m",
      "dim": "\033[2m", "off": "\033[0m"}


def _c(s, colour):
    if not sys.stdout.isatty():
        return s
    return f"{_C[colour]}{s}{_C['off']}"


def render(ans: Answer) -> str:
    lines = []
    body = []
    for seg in ans.segments:
        if seg.kind == "verbatim" and seg.matched:
            body.append(seg.text)
        elif seg.kind == "verbatim" and not seg.matched:
            body.append(_c(f"⚠{seg.text}⚠", "red"))
        else:
            body.append(_c(seg.text, "dim"))
    lines.append("".join(body))
    lines.append("")

    cert = ans.certificate
    status = {
        "grounded": _c("● GROUNDED", "green"),
        "repaired": _c("● GROUNDED (self-repaired)", "cyan"),
        "abstained": _c("● ABSTAINED (honest)", "yellow"),
        "blocked": _c("● BLOCKED (would not ship)", "red"),
    }.get(cert.grounding, cert.grounding)
    lines.append(_c("── faithfulness certificate ──", "bold"))
    lines.append(f"  status         {status}")
    if not cert.abstained:
        lines.append(f"  verbatim        {cert.verbatim_ratio:.0%} of the answer is her own words")
        lines.append(f"  claims          {cert.claims_supported} supported · "
                     f"{cert.claims_extrapolated} extrapolated · "
                     f"{_c(str(cert.claims_unsupported)+' unsupported', 'red' if cert.claims_unsupported else 'grey')}")
        if cert.fabricated_quotes:
            lines.append(f"  {_c('fabricated', 'red')}      {cert.fabricated_quotes} span(s) faked a quote")
    for n in cert.notes:
        lines.append(f"  {_c('note', 'yellow')}           {n}")
    if cert.sources:
        srcs = ", ".join(f"[{s['id']}·{s['kind']}]" for s in cert.sources)
        lines.append(f"  sources         {srcs}")
    return "\n".join(lines)

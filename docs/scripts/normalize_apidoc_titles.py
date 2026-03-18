#!/usr/bin/env python3
"""Remove sphinx-apidoc "package/module" suffixes from generated titles."""

from __future__ import annotations

import pathlib
import re

SUFFIX_PATTERN = re.compile(r"(?: package| module)$")


def normalize_file(path: pathlib.Path) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) < 2:
        return False

    title = lines[0].rstrip("\n")
    underline = lines[1].rstrip("\n")
    if not title or set(underline) != {"="}:
        return False

    normalized_title = SUFFIX_PATTERN.sub("", title)
    if normalized_title == title:
        return False

    lines[0] = f"{normalized_title}\n"
    lines[1] = f"{'=' * len(normalized_title)}\n"
    path.write_text("".join(lines), encoding="utf-8")
    return True


def main() -> None:
    base = pathlib.Path(__file__).resolve().parents[1] / "source" / "api" / "_generated"
    changed = 0
    for rst_file in sorted(base.glob("*.rst")):
        if normalize_file(rst_file):
            changed += 1
    print(f"Normalized {changed} file(s) in {base}")


if __name__ == "__main__":
    main()

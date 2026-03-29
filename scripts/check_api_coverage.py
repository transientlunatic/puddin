#!/usr/bin/env python3
"""
check_api_coverage.py — Verify that every public Puddin function listed in
api.toml is implemented in every binding layer.

Exit code 0 = all present.  Non-zero = at least one gap found.

Requires Python 3.11+ (tomllib is in the stdlib).
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


def ensure(failures: list[str], label: str, rel_path: str, needle: str) -> None:
    """Append a failure message if *needle* is not found in the file."""
    try:
        content = read(rel_path)
    except FileNotFoundError:
        failures.append(f"  FILE NOT FOUND  [{label}]  {rel_path}")
        return
    if needle not in content:
        failures.append(f"  MISSING  [{label}]  '{needle}'  in  {rel_path}")


def as_list(value: object) -> list[str]:
    """Return *value* as a list regardless of whether it is a str or list."""
    return [value] if isinstance(value, str) else list(value)


def main() -> int:
    manifest_path = ROOT / "api.toml"
    with manifest_path.open("rb") as fh:
        manifest = tomllib.load(fh)

    failures: list[str] = []

    for fn in manifest["function"]:
        name: str = fn["rust_name"]
        print(f"Checking: {name}")

        # ── Rust core ────────────────────────────────────────────────────────
        ensure(failures, "rust",
               "crates/puddin/src/binary.rs",
               f"pub fn {name}(")

        # ── C ABI  (library source + header) ────────────────────────────────
        for sym in as_list(fn.get("c_abi", [])):
            ensure(failures, "c_abi",
                   "bindings/julia/src/lib.rs",
                   f"{sym}(")
            ensure(failures, "c_header",
                   "bindings/julia/include/puddin.h",
                   f"{sym}(")

        # ── CLI ──────────────────────────────────────────────────────────────
        if "cli" in fn:
            ensure(failures, "cli",
                   "crates/puddin-cli/src/main.rs",
                   fn["cli"])

        # ── Python ───────────────────────────────────────────────────────────
        if "python" in fn:
            ensure(failures, "python",
                   "bindings/python/src/lib.rs",
                   f"wrap_pyfunction!({fn['python']}")

        # ── Julia ────────────────────────────────────────────────────────────
        if "julia" in fn:
            ensure(failures, "julia",
                   "bindings/julia/src/Puddin.jl",
                   f"function {fn['julia']}(")

        # ── R (source + NAMESPACE) ───────────────────────────────────────────
        if "r" in fn:
            ensure(failures, "r_source",
                   "bindings/r/R/puddin.R",
                   fn["r"])
            ensure(failures, "r_namespace",
                   "bindings/r/NAMESPACE",
                   fn["r"])

        # ── WASM TypeScript wrapper ──────────────────────────────────────────
        if "wasm" in fn:
            ensure(failures, "wasm",
                   "bindings/wasm/js/puddin.ts",
                   fn["wasm"])

    print()
    if failures:
        print(f"FAILED — {len(failures)} missing binding(s):\n")
        for msg in failures:
            print(msg)
        return 1

    print(f"OK — all {len(manifest['function'])} function(s) present in every layer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

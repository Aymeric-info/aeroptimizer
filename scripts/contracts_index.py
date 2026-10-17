#!/usr/bin/env python
"""Index CONTRACTS.md and keep it in sync with tests/ (ARCHITECTURE D-13).

    python scripts/contracts_index.py --check        # every contract has an existing test (CI gate)
    python scripts/contracts_index.py --gen-stubs    # create missing skipped test stubs (never overwrites)
    python scripts/contracts_index.py --list         # print the contract -> test table

A contract row of CONTRACTS.md looks like `| C-GRID-01 | text ... | test_grid::test_grid_validation |`.
The text cell may itself contain '|' (e.g. |D|), so the ID is the first cell and the test the LAST cell.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = ROOT / "CONTRACTS.md"
TESTS = ROOT / "tests"

ROW = re.compile(r"^\|\s*(C-[A-Z]+-\d+)\s*\|")
TEST_TOKEN = re.compile(r"`(test_\w+)::(test_\w+)`")

# First phase at which a contract's test is expected to pass (ARCHITECTURE §9).
PHASE_BY_PREFIX = {
    "CONV": 1, "GRID": 1, "MAT": 1, "ELEM": 1, "OP": 1, "BC": 1, "SOLVE": 1, "CFG": 1,
    "FILT": 2, "PROJ": 2, "SIMP": 2, "MASK": 2, "OPT": 2, "IO": 2, "DIFF": 2, "PERF": 2,
    "FLOW": 4, "AERO": 4, "OBJ": 4, "OVH": 5, "EXP": 5,
}
PHASE_OVERRIDES = {
    "C-ELEM-06": 4, "C-SOLVE-07": 2, "C-OBJ-04": 2, "C-CONV-07": 4, "C-CONV-09": 2,
    "C-DIFF-01": 6, "C-DIFF-02": 6,
}
INTEGRATION_FILES = {"test_mbb_benchmark", "test_gradcheck_pipeline", "test_pipeline"}


def parse_contracts() -> list[dict]:
    rows = []
    for line in CONTRACTS.read_text(encoding="utf-8").splitlines():
        m = ROW.match(line)
        if not m:
            continue
        body = line.rstrip().removesuffix("|")
        cells_head = body.split("|", 2)  # ['', ' C-ID ', ' text ... | test ']
        text_and_test = cells_head[2]
        text, _, test_cell = text_and_test.rpartition("|")
        tok = TEST_TOKEN.search(test_cell)
        rows.append({
            "id": m.group(1),
            "text": text.strip(),
            "test_cell": test_cell.strip(),
            "file": tok.group(1) if tok else None,
            "name": tok.group(2) if tok else None,
        })
    return rows


def phase_of(cid: str) -> int:
    if cid in PHASE_OVERRIDES:
        return PHASE_OVERRIDES[cid]
    return PHASE_BY_PREFIX[cid.split("-")[1]]


def test_path(file_stem: str) -> Path:
    sub = "integration" if file_stem in INTEGRATION_FILES else "unit"
    return TESTS / sub / f"{file_stem}.py"


def find_test_defs() -> dict[str, set[str]]:
    defs: dict[str, set[str]] = defaultdict(set)
    for py in TESTS.rglob("test_*.py"):
        for m in re.finditer(r"^def (test_\w+)\(", py.read_text(encoding="utf-8"), re.M):
            defs[py.stem].add(m.group(1))
    return defs


def cmd_check(rows: list[dict]) -> int:
    ids = [r["id"] for r in rows]
    dup = {i for i in ids if ids.count(i) > 1}
    defs = find_test_defs()
    errors = [f"duplicate contract id {i}" for i in sorted(dup)]
    no_pytest = []
    for r in rows:
        if r["file"] is None:
            no_pytest.append(r["id"])
        elif r["name"] not in defs.get(r["file"], set()):
            errors.append(f"{r['id']}: test {r['file']}::{r['name']} not found under tests/")
    print(f"{len(rows)} contracts, {len(rows) - len(no_pytest)} with a pytest test, "
          f"{len(no_pytest)} verified otherwise ({', '.join(no_pytest) or '-'})")
    for e in errors:
        print("ERROR:", e)
    return 1 if errors else 0


def cmd_list(rows: list[dict]) -> int:
    for r in rows:
        print(f"{r['id']:<11} phase {phase_of(r['id'])}  {r['file']}::{r['name']}" if r["file"]
              else f"{r['id']:<11} phase {phase_of(r['id'])}  ({r['test_cell']})")
    return 0


def cmd_gen_stubs(rows: list[dict]) -> int:
    grouped: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["file"]:
            grouped[r["file"]][r["name"]].append(r)
    existing = find_test_defs()
    created = 0
    for stem, tests in sorted(grouped.items()):
        path = test_path(stem)
        missing = {n: rs for n, rs in tests.items() if n not in existing.get(stem, set())}
        if not missing:
            continue
        new_file = not path.exists()
        chunks = []
        if new_file:
            module = stem.removeprefix("test_")
            chunks.append(f'"""Contract tests for `{module}` — stubs generated from CONTRACTS.md '
                          f'(scripts/contracts_index.py).\n\nReplace a stub body when its phase is reached.\n"""\n\n'
                          f"import pytest\n")
        for name, rs in missing.items():
            ids = ", ".join(r["id"] for r in rs)
            phase = min(phase_of(r["id"]) for r in rs)
            summary = rs[0]["text"].replace('"""', "'''")
            summary = (summary[:110] + "…") if len(summary) > 110 else summary
            chunks.append(
                f'\n\n@pytest.mark.skip(reason="phase {phase} — {ids}")\n'
                f"def {name}():\n"
                f'    r"""{ids} — {summary}"""\n'
                f'    raise NotImplementedError\n'
            )
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a" if not new_file else "w", encoding="utf-8") as fh:
            fh.write("".join(chunks))
        created += len(missing)
    print(f"created {created} stub(s)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true")
    g.add_argument("--gen-stubs", action="store_true")
    g.add_argument("--list", action="store_true")
    args = ap.parse_args()
    rows = parse_contracts()
    if not rows:
        print("no contract rows found in CONTRACTS.md", file=sys.stderr)
        return 2
    if args.check:
        return cmd_check(rows)
    if args.list:
        return cmd_list(rows)
    return cmd_gen_stubs(rows)


if __name__ == "__main__":
    raise SystemExit(main())

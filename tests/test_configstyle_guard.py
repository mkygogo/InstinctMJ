from __future__ import annotations

from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "instinct_mjlab"


def _iter_py_files():
  for path in SRC_ROOT.rglob("*.py"):
    if "__pycache__" in path.parts:
      continue
    yield path


def test_no_legacy_compat_imports() -> None:
  needle = "from instinct_mjlab.utils._compat import"
  offenders: list[str] = []
  for path in _iter_py_files():
    text = path.read_text(encoding="utf-8")
    if needle in text:
      offenders.append(str(path.relative_to(SRC_ROOT)))
  assert not offenders, (
    "Legacy compatibility import is forbidden. "
    f"Offenders: {offenders}"
  )


def test_no_legacy_decorator_usage() -> None:
  needle = "@" + "config" + "class"
  offenders: list[str] = []
  for path in _iter_py_files():
    text = path.read_text(encoding="utf-8")
    if needle in text:
      offenders.append(str(path.relative_to(SRC_ROOT)))
  assert not offenders, (
    "Legacy config decorator usage is forbidden. "
    f"Offenders: {offenders}"
  )


def test_no_legacy_decorator_token_by_import() -> None:
  needle = "utils._compat"
  total = 0
  for path in _iter_py_files():
    text = path.read_text(encoding="utf-8")
    total += text.count(needle)
  assert total == 0, f"Unexpected legacy compat token count: {total}"

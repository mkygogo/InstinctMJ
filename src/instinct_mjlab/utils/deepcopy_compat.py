"""Runtime compatibility patches for Python deepcopy behavior."""

from __future__ import annotations

import abc
import copy as _copy_module


def patch_deepcopy_for_abcmeta() -> None:
  """Patch ``copy.deepcopy`` dispatch for ABC internals on Python 3.13+.

  Python 3.13 introduces ``_abc._abc_data`` objects that are not picklable.
  Some deepcopy paths can still encounter these objects via dataclass defaults
  and class metadata snapshots. Treat them as atomic to avoid deepcopy errors.
  """

  dispatch = _copy_module._deepcopy_dispatch  # type: ignore[attr-defined]
  atomic = _copy_module._deepcopy_atomic  # type: ignore[attr-defined]

  if abc.ABCMeta not in dispatch:
    dispatch[abc.ABCMeta] = atomic

  abc_data_type = type(abc.ABC._abc_impl)
  if abc_data_type not in dispatch:
    dispatch[abc_data_type] = atomic


__all__ = ["patch_deepcopy_for_abcmeta"]

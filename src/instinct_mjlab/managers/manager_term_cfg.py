"""Configuration terms for different managers."""

from __future__ import annotations
from dataclasses import dataclass

import torch
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

@dataclass(kw_only=True)
class MultiRewardCfg:
    """Configuration for a reward group. Please inherit it if you want to define
    your own reward group so that the manager can recognize it.
    """

    pass

@dataclass(kw_only=True)
class DummyRewardCfg:
    """A placeholder for reward cfg."""

    pass

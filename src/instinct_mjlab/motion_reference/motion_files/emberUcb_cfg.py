from __future__ import annotations

from collections.abc import Callable  # noqa: F401

from typing import Literal

import torch

from instinct_mjlab.motion_reference.motion_reference_cfg import MotionBufferCfg

from .amass_motion_cfg import AmassMotionCfg
from .emberUcb import EmberUcb

class EmberUcbCfg(AmassMotionCfg):
    """Configuration for the EMBER UCB formatted motion data"""

    class_type: type = EmberUcb

    base_link_name: str = "pelvis"
    """ Which base_link is expected the G1 to be. Considering EmberLab in UCB retargetted G1 and stores all links poses
    in the world frame, it is possible to select the based frame when needed.
    """

    supported_file_endings = ["jpos.npz"]

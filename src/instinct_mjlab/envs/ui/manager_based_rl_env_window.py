from __future__ import annotations

from typing import TYPE_CHECKING

try:
  from mjlab.envs.ui import ManagerBasedRlEnvWindow as _ManagerBasedRlEnvWindow
except ModuleNotFoundError:
  class _ManagerBasedRlEnvWindow:
    def __init__(self, env, window_name: str = "mjlab"):
      self.env = env
      self.window_name = window_name
      self.ui_window_elements = {}

    def _visualize_manager(self, title: str, class_name: str):
      del title
      del class_name

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class InstinctLabRLEnvWindow(_ManagerBasedRlEnvWindow):
  """Window manager for the RL environment.

  On top of the manager-based RL environment window, this class adds
  more controls for InstinctLab-specific widgets.
  """

  def __init__(self, env: ManagerBasedRlEnv, window_name: str = "mjlab"):
    """Initialize the window.

    Args:
      env: The environment object.
      window_name: The name of the window. Defaults to "mjlab".
    """
    # initialize base window
    super().__init__(env, window_name)
    self.env = env
    self.window_name = window_name

    # add custom UI elements
    if (
      "main_vstack" in self.ui_window_elements
      and "debug_frame" in self.ui_window_elements
      and "debug_vstack" in self.ui_window_elements
    ):
      with self.ui_window_elements["main_vstack"]:
        with self.ui_window_elements["debug_frame"]:
          with self.ui_window_elements["debug_vstack"]:
            self._visualize_manager(title="Monitors", class_name="monitor_manager")

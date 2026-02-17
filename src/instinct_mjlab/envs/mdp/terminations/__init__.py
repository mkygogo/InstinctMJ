try:
  from .general import *  # noqa: F401,F403
  from .motion_reference import *  # noqa: F401,F403
except ModuleNotFoundError:
  from mjlab.envs.mdp.terminations import *  # noqa: F401,F403

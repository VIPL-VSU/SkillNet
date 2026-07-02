"""Compatibility alias for the SkillNet client package.

New SkillNet code should import :mod:`skillnet_client`.  This module is kept so
older evaluation snippets that import ``openpi_client`` continue to work.
"""

from skillnet_client import *  # noqa: F403
from skillnet_client import __version__

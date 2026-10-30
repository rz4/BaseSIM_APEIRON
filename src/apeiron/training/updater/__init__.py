"""Training updaters.

The only name a sibling layer imports from here is :class:`BaseUpdater`; the
concrete modes are reached through their own modules or through
:func:`apeiron.training.updater.create_updater.create_updater`.
"""

from apeiron.training.updater.base import BaseUpdater as BaseUpdater

__all__ = ["BaseUpdater"]

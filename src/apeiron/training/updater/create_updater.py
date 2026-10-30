"""Map ``cfg.continual_learning.update_mode`` onto an updater instance."""

from __future__ import annotations

from apeiron.config.configuration import Config
from apeiron.training.updater.base import BaseUpdater
from apeiron.training.updater.ewc import OnlineEWCUpdater
from apeiron.training.updater.jvp_reg import JVPRegUpdater
from apeiron.training.updater.kfac import OnlineKFACUpdater
from apeiron.training.updater.no_updater import NoUpdater

#: The complete set of selectable update modes.
UPDATERS = {
    "base": BaseUpdater,
    "none": NoUpdater,
    "ewc_online": OnlineEWCUpdater,
    "kfac_online": OnlineKFACUpdater,
    "jvp_reg": JVPRegUpdater,
}


def create_updater(cfg: Config, modelHarness) -> BaseUpdater:
    """Build the updater named by ``cfg.continual_learning.update_mode``.

    Parameters
    ----------
    cfg : Config
        The resolved run configuration.
    modelHarness : BaseModelHarness
        The harness whose model, criterion and data the updater drives.

    Returns
    -------
    BaseUpdater
        An instance of the class the mode selects.

    Raises
    ------
    NotImplementedError
        If the mode is not one of :data:`UPDATERS`.
    """
    mode = cfg.continual_learning.update_mode
    try:
        updater_cls = UPDATERS[mode]
    except KeyError:
        raise NotImplementedError(
            f"Unknown update_mode: {mode!r}. Expected one of "
            f"{sorted(UPDATERS)}."
        ) from None
    return updater_cls(cfg=cfg, modelHarness=modelHarness)

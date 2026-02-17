"""Helpers for migrating legacy class-style manager configs to mjlab dict configs."""

from __future__ import annotations
from dataclasses import dataclass

from mjlab.managers import ObservationTermCfg


class AttrDict(dict):
    """Dictionary with attribute-style access.

    This keeps legacy ``cfg.term`` style usage working while satisfying mjlab
    managers that expect dictionary configs.
    """

    def __getattribute__(self, name: str):
        if name == "__dict__":
            return self
        return super().__getattribute__(name)

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def manager_terms_to_dict(cfg_obj: object | dict | None) -> AttrDict:
    """Convert a class-style manager term container to a dict-like config."""

    if cfg_obj is None:
        return AttrDict()
    if isinstance(cfg_obj, AttrDict):
        return cfg_obj
    if isinstance(cfg_obj, dict):
        return AttrDict(cfg_obj)

    cfg_dict = AttrDict()
    for name, value in cfg_obj.__dict__.items():
        if name.startswith("_"):
            continue
        cfg_dict[name] = value
    return cfg_dict


def observation_terms_from_class(cfg_cls: type) -> dict[str, ObservationTermCfg | None]:
    """Collect observation terms declared as class attributes.

    Handles both plain class attributes and dataclass fields (created by @dataclass(kw_only=True)).
    When @dataclass(kw_only=True) processes mutable defaults (like ObsTermCfg instances), they
    become ``default_factory`` lambdas.  We call the factory to recover the value.
    """
    import dataclasses

    terms: dict[str, ObservationTermCfg | None] = {}
    # First, check plain class-level attributes
    for name, value in cfg_cls.__dict__.items():
        if name.startswith("_"):
            continue
        if isinstance(value, ObservationTermCfg) or value is None:
            terms[name] = value
    # Also check dataclass fields (populated by @dataclass(kw_only=True) / @dataclass)
    if dataclasses.is_dataclass(cfg_cls):
        for f in dataclasses.fields(cfg_cls):
            if f.name.startswith("_") or f.name in terms:
                continue
            # Resolve the default value
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:
                default = f.default_factory()
            else:
                continue
            if isinstance(default, ObservationTermCfg) or default is None:
                terms[f.name] = default
    return terms

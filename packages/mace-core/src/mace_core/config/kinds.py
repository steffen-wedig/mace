"""Config sections chosen by a ``kind`` field, open to plugins.

A direct subclass of :class:`KindedSection` is a *family*; a subclass of a
family with ``kind: Literal["name"] = "name"`` is one of its kinds, registered
when its class is defined. A field typed with the family accepts any of its
kinds and validates the rest with that kind's class.

Constraints that keep this short:

* A kind's name is the default of its ``kind`` field.
* Kinds are known once their module is imported. A plugin's modules are
  imported through the ``mace.plugins`` entry point group
  (:func:`~mace_core.plugins.load_plugins`), on the first kind not found.
* Configs are dumped through ``BaseConfig.to_resolved_dict`` and
  ``to_user_dict`` only, which keep a kind's own fields under a field typed
  with its family. A bare ``model_dump`` drops them.
"""

from __future__ import annotations

from typing import Any, ClassVar

from pydantic import GetJsonSchemaHandler, model_validator
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from mace_core.config.base import ConfigSection
from mace_core.plugins import load_plugins

__all__ = ["KindedSection"]


class KindedSection(ConfigSection):
    """The base of a family of config sections chosen by ``kind``.

    Attributes:
        kind: The kind's registered name.
        kinds: On a family, its kinds by name.
    """

    kind: str
    kinds: ClassVar[dict[str, type[KindedSection]]]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if KindedSection in cls.__bases__:
            cls.kinds = {}
            return
        name = cls.model_fields["kind"].default
        if name in cls.kinds:
            raise TypeError(
                f"kind {name!r} is defined twice: {cls.kinds[name].__qualname__} "
                f"and {cls.__qualname__}."
            )
        cls.kinds[name] = cls

    @model_validator(mode="wrap")
    @classmethod
    def _choose_kind(cls, data: Any, handler: Any) -> Any:
        """On a family, validate with the class ``data["kind"]`` names."""
        if KindedSection not in cls.__bases__ or not isinstance(data, dict):
            return handler(data)
        kind = data.get("kind")
        if kind not in cls.kinds:
            failed = load_plugins()
            if kind not in cls.kinds:
                raise ValueError(
                    f"{kind!r} is not a {cls.__name__} kind; the kinds are "
                    f"{sorted(cls.kinds)}."
                    + (f" Plugins that failed to import: {failed}." if failed else "")
                )
        return cls.kinds[kind].model_validate(data)

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """A family's schema is one of its kinds' schemas."""
        if KindedSection not in cls.__bases__:
            return handler(core_schema)
        kinds = cls.kinds.values()
        return {"oneOf": [handler(kind.__pydantic_core_schema__) for kind in kinds]}

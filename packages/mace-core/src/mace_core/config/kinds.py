"""Config sections that choose an implementation by a ``kind`` field.

A **family** is a direct subclass of :class:`KindedSection`; each **kind** is a
subclass of the family whose ``kind`` is a one-value ``Literal``::

    class NeighborListConfig(KindedSection):
        entry_point_group: ClassVar[str] = "mace.neighbor_list"

    class VesinConfig(NeighborListConfig):
        kind: Literal["vesin"] = "vesin"
        n_threads: int = Field(1, ge=0)

A field annotated with the family accepts any of its kinds:
``neighbor_list: NeighborListConfig``. The ``kind`` key picks the class, and that
class validates the rest, so an unknown key or a bad value fails when the config
is loaded, with the kind's own field names. The dump keeps the kind's fields
although the field is annotated with the family.

A kind defined in an imported module is registered when its class is defined.
A kind from another distribution is found through the family's
``entry_point_group``, whose entry named after the kind is the config class;
only that entry is imported, and only when a config names the kind. Config
modules must therefore stay light: pydantic, numpy, no framework and no backend.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from typing import Any, ClassVar, Literal, Self, cast, get_args, get_origin

from pydantic import (
    GetJsonSchemaHandler,
    SerializationInfo,
    SerializerFunctionWrapHandler,
    ValidatorFunctionWrapHandler,
    model_serializer,
    model_validator,
)
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema

from mace_core.config.base import ConfigSection
from mace_core.plugins import DuplicateRegistrationError, registered_entry_points

__all__ = ["KindedSection"]

# Set while a kind serializes itself; see `_dump_own_fields`.
_SERIALIZING_OWN_TYPE: ContextVar[bool] = ContextVar("own_type", default=False)


class KindedSection(ConfigSection):
    """The base of every config family whose members are chosen by ``kind``.

    Attributes:
        kind: The registered name of the kind. Each kind declares it as a
            one-value ``Literal`` with that value as its default.
        entry_point_group: Set on a family to let other distributions add
            kinds: the entry named after a kind is its config class.
    """

    kind: str
    entry_point_group: ClassVar[str | None] = None
    _kinds: ClassVar[dict[str, type[KindedSection]]]

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        if cls._is_family():
            cls._kinds = {}
            return
        kind = cls.kind_name()
        family = cls.family()
        seen = family._kinds.get(kind)
        if seen is not None and seen is not cls:
            raise DuplicateRegistrationError(
                f"{family.__name__} kind {kind!r} is defined twice: "
                f"{seen.__module__}.{seen.__qualname__} and "
                f"{cls.__module__}.{cls.__qualname__}."
            )
        family._kinds[kind] = cls

    @classmethod
    def _is_family(cls) -> bool:
        return KindedSection in cls.__bases__

    @classmethod
    def family(cls) -> type[KindedSection]:
        """The family this class belongs to; itself for a family."""
        for klass in cls.__mro__:
            if issubclass(klass, KindedSection) and KindedSection in klass.__bases__:
                return klass
        raise TypeError(f"{cls.__name__} is not in a KindedSection family")

    @classmethod
    def kind_name(cls) -> str:
        """The name a kind is registered under, read from its ``Literal``.

        Raises:
            TypeError: if ``kind`` is not a one-string ``Literal`` whose value
                is also its default.
        """
        annotation = cls.model_fields["kind"].annotation
        values = get_args(annotation) if get_origin(annotation) is Literal else ()
        if len(values) != 1 or not isinstance(values[0], str):
            raise TypeError(
                f"{cls.__name__}.kind must be a one-string Literal, such as "
                f'kind: Literal["vesin"] = "vesin"; it is {annotation!r}.'
            )
        if cls.model_fields["kind"].default != values[0]:
            raise TypeError(
                f"{cls.__name__}.kind must default to {values[0]!r}, so that "
                f"{cls.__name__}() needs no arguments."
            )
        return values[0]

    @classmethod
    def kinds(cls) -> list[str]:
        """Every kind of the family: defined here, or installed elsewhere."""
        family = cls.family()
        names = set(family._kinds)
        if family.entry_point_group:
            names |= set(registered_entry_points(family.entry_point_group))
        return sorted(names)

    @classmethod
    def kind_class(cls, kind: str) -> type[Self]:
        """The config class of ``kind``, importing it from its entry point if
        no imported module defined it.

        Raises:
            ValueError: if no such kind is registered or its module failed to
                import. Inside validation, pydantic reports it at the field.
        """
        family = cls.family()
        if kind not in family._kinds and family.entry_point_group:
            entry = registered_entry_points(family.entry_point_group).get(kind)
            if entry is not None:
                try:
                    entry.load()
                except Exception as failure:
                    raise ValueError(
                        f"the {family.__name__} kind {kind!r} is installed but "
                        f"its config did not import: {failure!r}"
                    ) from failure
        found = family._kinds.get(kind)
        if found is None:
            raise ValueError(
                f"{kind!r} is not a {family.__name__} kind. The kinds are "
                f"{cls.kinds()}."
            )
        return cast(type[Self], found)

    @classmethod
    def of(cls, kind: str, **fields: Any) -> Self:
        """The ``kind`` section with ``fields``, validated as a config file is."""
        return cls.model_validate({"kind": kind, **fields})

    @classmethod
    def __get_pydantic_json_schema__(
        cls, core_schema: CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        """A family's schema is one of its kinds, as far as they are imported.

        Kinds installed but not yet imported are missing, so a schema meant
        for an editor is written after the plugins are loaded. The serializer
        writes exactly the kind's fields, so both modes describe those fields.
        """
        if not cls._is_family():
            return handler(_without_serializer(core_schema))
        kinds = [cls._kinds[name] for name in sorted(cls._kinds)]
        return {
            "oneOf": [
                handler(_without_serializer(kind.__pydantic_core_schema__))
                for kind in kinds
            ],
            "title": cls.__name__,
        }

    @model_validator(mode="wrap")
    @classmethod
    def _choose_kind(cls, data: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """On a family, hand the data to the class its ``kind`` names."""
        if not cls._is_family() or isinstance(data, cls):
            return handler(data)
        if not isinstance(data, Mapping) or "kind" not in data:
            raise ValueError(f"a {cls.__name__} needs a kind, one of {cls.kinds()}.")
        return cls.kind_class(data["kind"]).model_validate(data)

    @model_serializer(mode="wrap")
    def _dump_own_fields(
        self, handler: SerializerFunctionWrapHandler, info: SerializationInfo
    ) -> Any:
        """Dump the kind's fields even under a field annotated with its family.

        Pydantic serializes by the declared type, which would drop every field
        the kind adds, so the resolved config would not load back. The kind's
        own serializer runs instead; the flag keeps that call from recursing,
        and is cleared for the sections nested inside it.
        """
        if _SERIALIZING_OWN_TYPE.get():
            token = _SERIALIZING_OWN_TYPE.set(False)
            try:
                return handler(self)
            finally:
                _SERIALIZING_OWN_TYPE.reset(token)
        token = _SERIALIZING_OWN_TYPE.set(True)
        try:
            return type(self).__pydantic_serializer__.to_python(
                self,
                mode=info.mode,
                by_alias=info.by_alias,
                exclude_unset=info.exclude_unset,
                exclude_defaults=info.exclude_defaults,
                exclude_none=info.exclude_none,
                round_trip=info.round_trip,
                context=info.context,
            )
        finally:
            _SERIALIZING_OWN_TYPE.reset(token)


def _without_serializer(schema: CoreSchema) -> CoreSchema:
    """``schema`` without ``_dump_own_fields``, which a schema cannot see into.

    The serializer sits on the model schema, under the validators wrapping it.
    """
    if schema["type"] == "model":
        return cast(
            CoreSchema, {k: v for k, v in schema.items() if k != "serialization"}
        )
    inner = schema.get("schema")
    if isinstance(inner, dict):
        return cast(CoreSchema, {**schema, "schema": _without_serializer(inner)})
    return schema

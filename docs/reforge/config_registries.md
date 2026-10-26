# Building objects from configs: one pattern for every swappable piece

*Design draft, 2026-10-26. Prototype: branch `reforge/data-3a-neighbors-registry`
on `steffen-wedig/mace`, on top of #1778 (`ca167914`).*

**Proposal.** Every swappable piece is a *family* of config sections chosen by
`kind`. A kind's config class says what can be set and is checked when the
config loads. A builder, registered separately per framework, turns that config
into an object. Both are registered when their module is imported. A plugin
is one module in the `mace.plugins` entry point group, so it adds a kind
without editing MACE.
The prototype ports the neighbour list to this pattern. All tests pass, and
`ty` and mypy both reject a builder registered with the wrong config or product.

## 1. The problem

Today the reforge builds objects from configs in three different ways, and the
plugin doc promises a fourth:

| Where | Config shape | Checked when | Built by | Can a plugin add a kind? |
| --- | --- | --- | --- | --- |
| `mace_torch/nn/embedding.py` (radial basis, distance transform) | One typed section per kind, `kind: Literal[...]`, closed union | When the config loads | `match` in mace-torch | No, the `match` must be edited |
| `mace_core/neighbors/backends.py` (#1778) | `kind: str` plus `settings: dict[str, Any]` | When the backend is built, possibly in a worker | Entry point, then `factory(**settings)` | Yes |
| `mace_core/kernels/registry.py` | Name only | n/a | Entry point, then `factory()` | Yes |
| `docs/reforge/extending_plugin.md` | Not specified | Not specified | `@register_*` decorators | Yes |

What goes wrong:

- **Settings are checked too late.** In #1778,
  `neighbor_list = {kind = "vesin", settings = {n_thread = 4}}` passes config
  loading and fails only when the backend is built, possibly inside a data-loader
  worker. The resolved config saved with the run holds an untyped dict.
- **Two layouts for one idea.** `distance_transform = {kind = "agnesi", exponent_q = ...}`
  is flat; `neighbor_list = {kind = "vesin", settings = {...}}` is nested.
- **Closed sets where plugins were promised.** The plugin doc lists transforms
  and embeddings as registrable, but `build_distance_transform` can't be
  extended.
- **The registry is copied.** `kernels/registry.py` and `neighbors/backends.py`
  are near-copies, and the electrostatics backends would add a third.
- **A config can't be shared across frameworks.** `build_radial_basis` ties the
  config to torch. A Bessel basis needs a torch builder and a jax builder for
  the same config section.

## 2. Goals

- **Plugins add kinds**, both the config class and the builder, without
  editing MACE.
- **Fail early.**
  - A malformed registration fails when its class is defined.
  - An unknown kind, an unknown key or a bad value fails when the config
    loads, never in a worker.
- **Precise types.**
  - A builder takes its own concrete config class.
  - `build` returns the family's protocol, not `Any`.
  - Pairing a builder with the wrong config, or with something that isn't the
    product, is a type error.
- **Validation imports no backend.** Checking a config never imports vesin,
  torch, warp or CUDA, so the parent process is clean before workers fork.
- **No silent substitution.** A named kind that can't be built raises; nothing
  falls back to another backend.
- **Little syntax.** A plugin writes one config class, one builder, one
  `register` line and one entry point. Configs stay flat: a section with a
  `kind` key and that kind's fields, nothing nested under it.

Not goals: a plugin framework, dependency injection, or a config hierarchy
deeper than "a section with a `kind`". Sets that are closed for good, like the
single cutoff envelope, keep a plain constructor.

## 3. Design

### Constraints

A few rules keep the machinery small:

| Constraint | Saves |
| --- | --- |
| A family is a direct subclass of `KindedSection`; a kind is a subclass with `kind: Literal["name"] = "name"`, and its name is that default | Any declaration syntax beyond a normal subclass |
| Kinds and builders are registered when their module is imported. Plugins are modules in the one `mace.plugins` group (as `extending_plugin.md` already plans), imported once on the first kind that isn't found | Per-kind config entry points and per-kind lazy loading |
| Backend modules must be cheap to import: a heavy library (vesin, CUDA, warp) is imported when the backend is built, not at module level | Lazy builder loading, failure caching, builder entry points |
| Configs are dumped only through `BaseConfig.to_resolved_dict` and `to_user_dict`, which pass pydantic's `serialize_as_any=True` | A custom serializer to keep a kind's fields under a field typed with its family |
| Builders are statically typed | A runtime check on what they return |

The cost is two rules for plugin authors: import heavy libraries lazily, and
never `model_dump` a config directly. A test in each backend's suite can check
the first by importing the module in a fresh process.

### What a plugin writes

One module, listed under `mace.plugins`:

```python
class SkinConfig(NeighborListConfig):
    kind: Literal["skin"] = "skin"
    skin: float = Field(0.0, ge=0.0)

class SkinBackend:
    def __init__(self, config: SkinConfig) -> None: ...   # heavy imports go here
    def search(self, positions, cutoff, geometry): ...

NEIGHBOR_BACKENDS.register(SkinConfig, SkinBackend)
```

```toml
[project.entry-points."mace.plugins"]
skin = "skin_plugin"
```

The user's config stays flat: `neighbor_list = {kind = "skin", skin = 0.5}`.
Built-in kinds are written the same way; they are registered when
`mace_core.neighbors` is imported.

### What MACE defines per family

```python
class NeighborListConfig(KindedSection): ...             # the family

NEIGHBOR_BACKENDS: Builders[NeighborListConfig, NeighborListBackend, []] = Builders(
    NeighborListConfig)

class RunConfig(BaseConfig):
    neighbor_list: NeighborListConfig | None = None      # a plain annotation

backend = NEIGHBOR_BACKENDS.build(config.neighbor_list)  # typed NeighborListBackend
```

The third type parameter is what a builder takes besides its config. For a
radial basis it would be `[float]` (`r_max`), so `RADIAL_BASES.build(config,
r_max)` is checked statically too. A config class can have one `Builders` per
framework, so a Bessel basis can get a torch builder and a jax builder.

### The two generic pieces

- **`KindedSection`** (`mace_core/config/kinds.py`, 83 lines with
  docstrings):
  - registers kinds when their class is defined;
  - a field typed with a family picks the kind's class from `kind` and lets it
    validate the rest;
  - its JSON schema is a `oneOf` of the family's kinds.
- **`Builders`** (`mace_core/plugins.py`, 92 lines): a typed
  `register(config, builder)`, `build(config, *context)`, and `load_plugins()`.

They replace #1778's registry and are meant to replace `kernels/registry.py`
and the `match` in `embedding.py`.

## 4. Lifecycle and failures

- **When a class is defined:** a kind name used twice in a family raises
  `TypeError`.
- **When `register` is called:**
  - a config from another family raises `TypeError`;
  - a second builder for one config class raises `TypeError`.
- **When the config loads:**
  - an unknown or missing kind is a `ValidationError` at the field, listing the
    kinds and naming any plugin that failed to import;
  - an unknown key or a bad value is reported at the field
    (`neighbor_list.n_thread`).
- **When an object is built:**
  - a kind with no builder in this framework raises `NotAvailableError`;
  - a missing library raises `ImportError` naming the extra to install.
  - Nothing is substituted.

Validation imports no backend library: a test checks that `vesin`, `torch` and
`matscipy` are absent after importing `mace_core.neighbors` and validating a
vesin config. Workers receive the config and build their own backend; a test
does this in a spawned process and gets the same edges.

## 5. Type checking

All four mistakes below are caught by both `ty` 0.0.77 and mypy, run against
the real `NEIGHBOR_BACKENDS`:

| Mistake | Checker's verdict |
| --- | --- |
| `register(VesinConfig, SkinBackend)` | Builder takes `SkinConfig`, not `VesinConfig` |
| `register(SkinConfig, NotABackend)` | `NotABackend` is not a `NeighborListBackend` (no `search`) |
| `NEIGHBOR_BACKENDS.build(cfg, 3.0)` | Too many arguments: this family takes no context |
| `NEIGHBOR_BACKENDS.build({"kind": "skin"})` | A dict is not a `NeighborListConfig` |

`build` returns `NeighborListBackend`. Two things are runtime-only:

- **That a config belongs to the registry's family.** This would need a type
  variable bounded by another type variable, which Python can't express. It is
  a one-line `issubclass` check in `register`.
- **Looking a kind up by name.** `Family.kinds[name]` is typed with the base
  class. Code that needs the concrete type validates a dict instead.

## 6. The prototype

Branch `reforge/data-3a-neighbors-registry` on `steffen-wedig/mace`, on top of
#1778's `ca167914`.

- **New:**
  - `mace_core/plugins.py` (`Builders`, `load_plugins`, `NotAvailableError`);
  - `mace_core/config/kinds.py` (`KindedSection`);
  - `mace_core/neighbors/config.py` (`NeighborListConfig`, plus
    `BruteForceConfig`, `MatscipyConfig` and `VesinConfig(n_threads)`).
- **Changed:**
  - `BaseConfig`'s two dump methods pass `serialize_as_any=True`.
  - `neighbors/backends.py` loses its registry.
  - Each backend takes its config, imports its library in `__init__`, and ends
    with one `register` line.
  - The `mace.neighbor_backends` entry points are gone.
- **Removed from #1778's API:** `settings`, `get_neighbor_backend`,
  `available_neighbor_backends`, and the two neighbour-specific error types.
  `resolve_neighbor_backend(config, default)` and the per-framework defaults
  (vesin for torch, matscipy for jax) are unchanged.
- **Results:** 1645 passed, 4 skipped, 13 xfailed across mace-core,
  mace-torch, mace-jax, architecture, and the neighbour parity against legacy.
  `ruff`, `ty` and `lint-imports` (5 contracts) are clean.

## 7. How it compares with the closed union in `embedding.py`

| | Closed union (`Annotated[A \| B, Field(discriminator="kind")]`) | `KindedSection` family |
| --- | --- | --- |
| A plugin can add a kind | No | Yes |
| Error location | `radial_basis.bessel.num_basis` | `neighbor_list.n_threads`, the same as a plain section |
| JSON schema | Fixed when the code is written | The kinds imported when the schema is generated; write editor schemas after loading plugins |
| Static type of the field | The union | The family base |
| Machinery | None (pydantic built-in) | One validator and one schema hook, written once |

Moving `embedding.py` over is mechanical:
- each union becomes a family;
- the `match` becomes a `Builders[RadialBasisConfig, RadialBasis, [float]]` in
  mace-torch, with a jax one alongside it later.

The config files don't change. Its tests pin the error location under the tag,
which would move to the plain field path.

## 8. Open questions for the meeting

1. **Model state versus run config.** Neighbour and kernel backends don't
   change the numbers. Should their sections live in the run config and stay
   out of model metadata, so that loading a checkpoint never needs vesin or
   cuequivariance installed? Model families (radial basis, transforms,
   embeddings) must be registered for a checkpoint to load.
2. **Where the families live.** One `mace_core/registries.py`, as
   `target_layout.md` suggests, or each family next to its feature, as in the
   prototype (`neighbors/config.py`)?
3. **Import-light backend modules** is a rule for plugin authors. Should it be
   enforced by a shared conformance test, the way kernel backends have one?
4. **Order of migration:** `kernels/registry.py` (no configs yet),
   `embedding.py` (changes where errors are reported), then electrostatics and
   the `@register_*` registries in the plugin doc. This needs agreeing before
   the electrostatics backend lands, or it will pick its own pattern.
5. **Edge order across backends.** #1778 still leaves it unspecified, and
   vesin is now mace-torch's default. Should `get_neighborhood` sort edges
   into one canonical order, so any backend reproduces the same graph exactly?

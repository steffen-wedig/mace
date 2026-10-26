# Building objects from configs: one pattern for every swappable piece

*Design draft, 2026-10-26. Prototype: branch `reforge/data-3a-neighbors-registry`
on `steffen-wedig/mace`, on top of #1778 (`ca167914`).*

**Proposal.** Every swappable piece is a *family* of config sections chosen by
`kind`. A kind's config class says what can be set and is checked when the
config loads. A builder, registered separately per framework, turns that config
into an object, and its module is imported only when the kind is built. Both
are found through entry points, so a plugin adds a kind without editing MACE.
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
  `register` line and two entry points. Configs stay flat: a section with a
  `kind` key and that kind's fields, nothing nested under it.

Not goals: a plugin framework, dependency injection, or a config hierarchy
deeper than "a section with a `kind`". Sets that are closed for good, like the
single cutoff envelope, keep a plain constructor.

## 3. Design

### The two pieces

| | Config class | Builder |
| --- | --- | --- |
| Says | What can be set for a kind | How a kind becomes an object in one framework |
| Lives in | `mace_core` or the plugin's light config module (pydantic only) | The framework package, or `mace_core` for numpy builders |
| Registered | When its class is defined, as a subclass of the family | By `Builders.register(Config, builder)` in the builder's module |
| Found through | The family's entry point group (e.g. `mace.neighbor_list`), entry named after the kind | The registry's group (e.g. `mace.neighbor_backends`), entry named after the kind, pointing to the module |
| Imported | When a config names the kind, or when its module is imported | When the kind is built, in the process that builds it |

Keeping them apart is what lets one config class have a numpy, a torch and a
jax builder, and what keeps validation light. `target_layout.md` already plans
this split ("specs/names only; torch classes are registered in mace_torch").

### What a plugin writes

```python
# skin_config.py: light, imports pydantic only
class SkinConfig(NeighborListConfig):
    kind: Literal["skin"] = "skin"
    skin: float = Field(0.0, ge=0.0)

# skin_backend.py: imported only when "skin" is built
class SkinBackend:
    def __init__(self, config: SkinConfig) -> None: ...
    def search(self, positions, cutoff, geometry): ...

NEIGHBOR_BACKENDS.register(SkinConfig, SkinBackend)
```

```toml
[project.entry-points."mace.neighbor_list"]
skin = "skin_config:SkinConfig"
[project.entry-points."mace.neighbor_backends"]
skin = "skin_backend"
```

The user's config stays flat: `neighbor_list = {kind = "skin", skin = 0.5}`.
Built-in kinds work the same way. Their config classes sit next to the family,
so they are registered when the family is imported and need no
`mace.neighbor_list` entry.

### What MACE defines per family

```python
class NeighborListConfig(KindedSection):                 # the family
    entry_point_group: ClassVar[str | None] = "mace.neighbor_list"

NEIGHBOR_BACKENDS = Builders[NeighborListConfig, NeighborListBackend, []](
    NeighborListConfig, NeighborListBackend, group="mace.neighbor_backends")

class RunConfig(BaseConfig):
    neighbor_list: NeighborListConfig | None = None      # plain annotation, no Annotated[...]

backend = NEIGHBOR_BACKENDS.build(config.neighbor_list)  # typed NeighborListBackend
```

The third type parameter is what a builder needs beyond its config. For a radial
basis it would be `[float]` (`r_max`), so `RADIAL_BASES.build(config, r_max)` is
checked statically too.

### The two generic pieces

- **`KindedSection`** (`mace_core/config/kinds.py`, about 230 lines with
  docstrings) is a `ConfigSection` whose direct subclasses are families.
  - **Registration:** a subclass whose `kind` is a one-string `Literal`
    registers itself in its family.
  - **Validation:** a field typed with the family reads `kind`, picks the
    class, and lets that class validate the rest. If the kind isn't imported
    yet, it loads only the entry point of that one kind.
  - **Saving:** a serializer hook writes the kind's own fields. Pydantic would
    otherwise serialize by the declared family type and silently drop
    `n_threads`, so the resolved config wouldn't load back.
  - **JSON schema:** a schema hook lists every imported kind as a `oneOf`.
- **`Builders`** (`mace_core/plugins.py`, about 270 lines with docstrings) maps
  config classes to builders for one family in one framework.
  - **Methods:** `register(config, builder)`, `build(config, *context)`, and
    `available()` for listing.
  - **Loading:** the builder is imported from its entry point on the first
    build of its kind.
  - **Caching:** the entry-point metadata is read once per process.
  - **Failures:** a failed import is recorded, so a broken backend is
    re-raised rather than retried on every batch.
  - It would replace both `kernels/registry.py` and the registry in
    `neighbors/backends.py`.

## 4. Lifecycle and failures

```
startup / import   config load (main process)      build (worker or main)
-----------------  ------------------------------  ------------------------------
class defined:     kind -> config class;           type(config) -> builder;
  bad Literal,       entry point of that one kind    entry point of that one kind
  duplicate kind     if not yet imported             if not yet imported
  -> TypeError     unknown kind / key / value      no builder, import failed,
                     -> ValidationError at the       wrong product
                     field (lists the kinds)         -> NotAvailableError, cached
```

- **Once per process.** The first `build` of a kind imports its builder's
  module, whose `register` call runs; after that it is a dict lookup.
  - Forked workers inherit whatever the parent already loaded.
  - Spawned workers each load the kind on their first build.
  - Workers receive the config, which pickles trivially, and build their own
    backend.
- **Validation never imports a builder.** A test checks this in a fresh
  process: after validating `vesin` and `matscipy` configs, `vesin`, `torch`,
  `matscipy` and the vesin backend module are all absent from `sys.modules`.
- **Name collisions are refused.**
  - Two distributions registering different values under one entry-point name
    → `DuplicateRegistrationError`.
  - Two config classes claiming one kind → refused when the second class is
    defined.
  - A second, different builder for one config class → refused.
  - Registering the same object twice is a no-op.

## 5. Type checking

All four mistakes below are caught by both `ty` 0.0.77 and mypy, run against
the real `NEIGHBOR_BACKENDS`:

| Mistake | Checker's verdict |
| --- | --- |
| `register(VesinConfig, SkinBackend)` | Builder takes `SkinConfig`, not `VesinConfig` |
| `register(SkinConfig, NotABackend)` | `NotABackend` is not a `NeighborListBackend` (no `search`) |
| `NEIGHBOR_BACKENDS.build(cfg, 3.0)` | Too many arguments: this family takes no context |
| `NEIGHBOR_BACKENDS.build({"kind": "skin"})` | A dict is not a `NeighborListConfig` |

`build` returns `NeighborListBackend`. Three things are checked only at
runtime, each at registration or config load:

- **That the config class is in the registry's family.** This would need a
  method type variable bounded by the class's type variable, which Python's
  type system doesn't allow.
- **The lookup inside `build`.** It returns `Callable[..., Product]`, because
  a mapping from a type to a value that depends on it can't be expressed. The
  `...` stays inside the registry; callers see a precise signature.
- **That the product implements the protocol.** It is checked with
  `isinstance` when the protocol is `runtime_checkable`, which catches dynamic
  registrations the type checker never sees.

Registration is a plain call, not a decorator. A class decorator that returns
its argument unchanged can't also check the pairing.

## 6. The prototype

Branch `reforge/data-3a-neighbors-registry` on `steffen-wedig/mace`, one commit
on top of #1778's `ca167914`.

- **New:**
  - `mace_core/plugins.py` (`Builders`, the entry-point cache, the two error
    types);
  - `mace_core/config/kinds.py` (`KindedSection`);
  - `mace_core/neighbors/config.py` (`NeighborListConfig`, plus
    `BruteForceConfig`, `MatscipyConfig` and `VesinConfig(n_threads)`).
- **Changed:**
  - `neighbors/backends.py` loses its own registry (−218 lines).
  - Each backend takes its config and ends with one `register` line.
  - The `mace.neighbor_backends` entry points name modules instead of classes.
- **Removed from #1778's API:**
  - `settings`, `get_neighbor_backend` and `available_neighbor_backends`;
  - `NeighborBackendNotAvailableError` and `DuplicateNeighborBackendError`,
    replaced by the generic `NotAvailableError` and
    `DuplicateRegistrationError`.
  - `resolve_neighbor_backend(config, default)` and the per-framework defaults
    (vesin for torch, matscipy for jax) are unchanged.
- **Results:**
  - Tests: 1655 passed, 4 skipped, 13 xfailed. This covers mace-core,
    mace-torch, mace-jax, architecture, and the neighbour parity against
    legacy, which is exact for matscipy and edge-set equal for every backend.
  - `ruff`, `ty` and `lint-imports` (5 contracts) are clean.
- **New tests cover:**
  - typos and bad values rejected at load, reported at the field;
  - unknown or missing kinds, with the list of kinds;
  - saving and reloading under a family annotation;
  - the JSON schema in both modes;
  - validation importing no backend (in a fresh process);
  - a plugin kind validated and built through entry points alone, with only
    its config module imported during validation;
  - a backend that fails to import, reported with its reason, cached, and not
    replaced;
  - a kind with no builder; a module that registers nothing; a product that
    isn't a backend;
  - duplicate entry points, kinds and builders; malformed `kind` declarations;
  - every backend pickling;
  - a config built in a spawned worker giving the same edges.

## 7. How it compares with the closed union in `embedding.py`

| | Closed union (`Annotated[A \| B, Field(discriminator="kind")]`) | `KindedSection` family |
| --- | --- | --- |
| A plugin can add a kind | No | Yes |
| Error location | `radial_basis.bessel.num_basis` | `neighbor_list.n_threads`, the same as a plain section |
| JSON schema | Fixed when the code is written | The kinds imported when the schema is generated; write editor schemas after loading plugins |
| Static type of the field | The union | The family base |
| Machinery | None (pydantic built-in) | One validator, one serializer and one schema hook, written once |

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
3. **Two entry points per plugin kind** (config and builder): is that
   acceptable? Folding them into one would mean either importing the builder
   during validation, or a string reference to the builder that type checkers
   can't follow.
4. **Order of migration:** `kernels/registry.py` (no configs yet),
   `embedding.py` (changes where errors are reported), then electrostatics and
   the `@register_*` registries in the plugin doc. This needs agreeing before
   the electrostatics backend lands, or it will pick its own pattern.
5. **Edge order across backends.** #1778 still leaves it unspecified, and
   vesin is now mace-torch's default. Should `get_neighborhood` sort edges
   into one canonical order, so any backend reproduces the same graph exactly?

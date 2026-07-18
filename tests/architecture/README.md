# Architecture policy

This suite checks dependency direction and responsibility boundaries in production
code. It builds an AST-only graph of `nlpo_toolkit/**/*.py`; importing or executing
the production package is deliberately unnecessary. `scripts/` is excluded because
it contains developer entry points rather than installed production modules.

The inward direction is: foundation/contracts → pure domain → application services
and infrastructure adapters → composition root and CLI. Pure domain may depend on
contracts but cannot perform filesystem/process/terminal I/O. Application services
receive ports and cannot import CLI or production composition. Infrastructure
implements ports without depending on application commands. Only CLI adapters may
consume production dependency factories from the composition root.

Module cycles and cycles between the package groups in `policy.py` are forbidden.
Function-local and `TYPE_CHECKING` imports count as dependencies. Literal dynamic
imports become graph edges; non-literal targets require a narrow, reasoned allowance
in the central policy. Cross-module private imports are forbidden.

Architecture tests answer “may this responsibility depend on or perform that?”. Unit
and integration tests answer “does this value or behavior work?”. Consequently this
suite does not normally freeze filenames, function names, signatures, dataclass field
sets, `__init__.py` emptiness, serialization content, or historical rename state.

Architecture tests protect concrete dependency direction; module-role coverage is
not required to be exhaustive. An unclassified production module is diagnostic
information, not an architecture violation. Existing classifications must remain
well formed and non-conflicting, but a new classification or layer should be added
only when a concrete dependency problem requires it.

Classification coverage can be inspected on demand without affecting CI:

```bash
python -m tests.architecture.module_role_report
```

The report uses the same production graph and role selectors as the tests, lists
unclassified modules in stable order, and exits successfully whether or not any are
present.

Frozen value and result models own deeply read-only collections. Ordered values are
normalized to tuples, set-valued fields to frozensets, and mappings to defensive
read-only copies. `Counter` is an execution-time accumulator only; result boundaries
expose immutable count mappings. Mutable analysis-cache collectors are converted to
snapshots before entering final results. Serializers alone convert these collections
back to ordinary dictionaries and lists. This guarantee applies to owned configuration,
DTO, domain, result, and metadata values—not to opaque injected backends, callables,
repositories, or context managers.

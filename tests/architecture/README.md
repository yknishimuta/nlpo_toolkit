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

When adding a module, classify its role in `policy.py` and add its stable role prefix
to the smallest applicable group. Prefer a general dependency or source rule over a
symbol blacklist. Add an allowlist entry only when the dependency is intentional,
narrowly scoped, cannot be expressed by the normal direction, and has a durable
reason; migration-only exceptions are not accepted.


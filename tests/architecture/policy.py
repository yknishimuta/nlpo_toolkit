from __future__ import annotations

from .support.rules import DependencyRule, DynamicImportAllowance


CA = "nlpo_toolkit.corpus_analysis"
CLI = f"{CA}.cli"
STANDALONE_CLI_MODULES = (
    "nlpo_toolkit.latin.cleaners.run_clean_corpus",
    "nlpo_toolkit.latin.latin_wordlist.build_latin_wordlist",
)
COMPOSITION = f"{CA}.composition"
PORTS = f"{CA}.ports"

APPLICATION_MODULES = (
    f"{CA}.analysis_orchestration",
    f"{CA}.count_command",
    f"{CA}.dry_run",
    f"{CA}.execution_session",
    f"{CA}.features.service",
    f"{CA}.ngram",
    f"{CA}.concordance",
    f"{CA}.planning.build",
    f"{CA}.planning.resolve",
    f"{CA}.preprocessing",
    f"{CA}.reporting.service",
    "nlpo_toolkit.comparison.services",
    "nlpo_toolkit.latin.cleaners.service",
)

PURE_MODULES = (
    "nlpo_toolkit.comparison.config",
    "nlpo_toolkit.comparison.engine",
    "nlpo_toolkit.comparison.metrics",
    "nlpo_toolkit.comparison.results",
    f"{CA}.analysis_results",
    f"{CA}.features.engine",
    f"{CA}.features.filtering",
    f"{CA}.features.lexical",
    f"{CA}.features.mfw",
    f"{CA}.features.models",
    f"{CA}.features.upos",
    f"{CA}.partition_models",
    f"{CA}.partition_validation",
    f"{CA}.planning.models",
    f"{CA}.planning.validate",
    f"{CA}.postprocessing",
    f"{CA}.reporting.models",
    f"{CA}.reporting.summary",
    f"{CA}.token_artifact.codec",
    f"{CA}.token_artifact.integrity",
    f"{CA}.token_artifact.schema",
    f"{CA}.token_sequences",
)

INFRASTRUCTURE_MODULES = (
    "nlpo_toolkit.backends",
    f"{CA}.analysis_cache",
    f"{CA}.archive",
    f"{CA}.artifacts.publication",
    f"{CA}.artifacts.writers",
    f"{CA}.cache_storage",
    f"{CA}.token_artifact.reader",
    f"{CA}.token_artifact.writer",
)

DEPENDENCY_RULES = (
    DependencyRule(
        "non-cli-no-cli",
        ("nlpo_toolkit",),
        (CLI,),
        excluded_source_prefixes=(CLI,),
        explanation="Production modules outside the CLI adapter must not depend on CLI modules.",
    ),
    DependencyRule(
        "standalone-cli-inward-only",
        ("nlpo_toolkit",),
        STANDALONE_CLI_MODULES,
        excluded_source_prefixes=STANDALONE_CLI_MODULES,
        explanation="Standalone CLI adapters may call application code, but production code must not depend on those adapters.",
    ),
    DependencyRule(
        "application-no-composition",
        APPLICATION_MODULES,
        (COMPOSITION,),
        explanation="Application services receive dependencies through ports; they do not assemble production implementations.",
    ),
    DependencyRule(
        "composition-consumed-only-by-cli",
        ("nlpo_toolkit",),
        (COMPOSITION,),
        excluded_source_prefixes=(CLI, COMPOSITION),
        explanation="Only CLI bootstrap modules may consume the production composition root.",
    ),
    DependencyRule(
        "application-domain-no-backends",
        (CA,),
        ("nlpo_toolkit.backends",),
        excluded_source_prefixes=(CLI, COMPOSITION),
        explanation="Application and reporting code depend on NLP contracts, not concrete backend implementations.",
    ),
    DependencyRule(
        "ports-inward-only",
        (PORTS,),
        (CLI, COMPOSITION, "nlpo_toolkit.backends", "nlpo_toolkit.latin.cleaners", f"{CA}.archive", f"{CA}.artifacts.writers"),
        explanation="Ports contain contracts and immutable dependency containers, never concrete adapters.",
    ),
    DependencyRule(
        "infrastructure-no-composition-or-cli",
        INFRASTRUCTURE_MODULES,
        (CLI, COMPOSITION),
        explanation="Infrastructure implements inward-facing contracts and must not know the bootstrap or CLI.",
    ),
    DependencyRule(
        "backend-domain-independent",
        ("nlpo_toolkit.backends",),
        (CA,),
        explanation="NLP backends implement NLP contracts and remain independent of corpus-analysis applications.",
    ),
    DependencyRule(
        "comparison-independent",
        ("nlpo_toolkit.comparison",),
        (CA,),
        explanation="The comparison package is reusable and independent of corpus-analysis orchestration.",
    ),
    DependencyRule(
        "comparison-config-inward",
        ("nlpo_toolkit.comparison.config",),
        ("nlpo_toolkit.comparison.engine", "nlpo_toolkit.comparison.results", "nlpo_toolkit.comparison.services"),
        explanation="Comparison configuration is a lower-level contract.",
    ),
    DependencyRule(
        "comparison-core-no-services",
        ("nlpo_toolkit.comparison.engine", "nlpo_toolkit.comparison.metrics", "nlpo_toolkit.comparison.results"),
        ("nlpo_toolkit.comparison.config", "nlpo_toolkit.comparison.frequency_io", "nlpo_toolkit.comparison.services", CLI),
        explanation="Comparison calculations and results do not depend on configuration, adapters, or services.",
    ),
    DependencyRule(
        "planning-models-inward",
        (f"{CA}.planning.models",),
        (f"{CA}.planning.build", f"{CA}.planning.resolve", f"{CA}.planning.validate", f"{CA}.preprocessing"),
        explanation="Planning models are independent values shared by planning stages.",
    ),
    DependencyRule(
        "planning-validate-pure",
        (f"{CA}.planning.validate",),
        (f"{CA}.planning.build", f"{CA}.planning.resolve", f"{CA}.preprocessing", "nlpo_toolkit.configuration"),
        explanation="Validation checks plans without resolving files or executing preprocessing.",
    ),
    DependencyRule(
        "planning-build-no-later-stage",
        (f"{CA}.planning.build",),
        (f"{CA}.planning.resolve", f"{CA}.preprocessing"),
        explanation="Plan construction does not execute later resolution or preprocessing stages.",
    ),
    DependencyRule(
        "planning-resolve-no-build",
        (f"{CA}.planning.resolve",),
        (f"{CA}.planning.build", "nlpo_toolkit.configuration"),
        explanation="Resolution consumes a plan and never rebuilds or reloads its configuration.",
    ),
    DependencyRule(
        "token-artifact-core-inward",
        (f"{CA}.token_artifact.schema", f"{CA}.token_artifact.codec"),
        (f"{CA}.token_artifact.reader", f"{CA}.token_artifact.writer", f"{CA}.token_artifact.validation"),
        explanation="Token artifact schemas and codecs are independent of I/O adapters and validation workflows.",
    ),
    DependencyRule(
        "token-artifact-reader-writer-independent",
        (f"{CA}.token_artifact.reader",),
        (f"{CA}.token_artifact.writer",),
        explanation="Reading and publication are separate adapters.",
    ),
    DependencyRule(
        "token-artifact-writer-reader-independent",
        (f"{CA}.token_artifact.writer",),
        (f"{CA}.token_artifact.reader",),
        explanation="Reading and publication are separate adapters.",
    ),
    DependencyRule(
        "token-sequences-reusable",
        (f"{CA}.token_sequences",),
        (f"{CA}.ngram", f"{CA}.concordance", CLI, f"{CA}.token_artifact.reader", f"{CA}.corpus"),
        explanation="Token sequence consumers depend on the common sequence domain, never the reverse.",
    ),
    DependencyRule(
        "artifact-models-inward",
        (f"{CA}.artifacts.models", f"{CA}.artifacts.planning"),
        (f"{CA}.artifacts.writers", f"{CA}.artifacts.publication", f"{CA}.reporting"),
        explanation="Artifact inventory and planning are independent of publication and reporting.",
    ),
    DependencyRule(
        "analysis-result-inward",
        (f"{CA}.analysis_results",),
        (f"{CA}.artifacts.writers", f"{CA}.reporting"),
        explanation="Analysis results do not depend on their publication adapters.",
    ),
    DependencyRule(
        "reporting-models-inward",
        (f"{CA}.reporting.models", f"{CA}.reporting.metadata", f"{CA}.reporting.summary"),
        (f"{CA}.reporting.service", f"{CA}.artifacts.writers"),
        explanation="Reporting values, metadata construction, and rendering are independent of publication orchestration.",
    ),
)

# There are currently no justified dynamic-import exceptions. Entries must name a
# narrow source/target and explain why static composition cannot be used.
DYNAMIC_IMPORT_ALLOWANCES: tuple[DynamicImportAllowance, ...] = ()

SERIALIZATION_BOUNDARY_MODULES = (
    "nlpo_toolkit.serialization",
    "nlpo_toolkit.configuration.yaml_loader",
    f"{CA}.config.parser",
    f"{CA}.config.serializer",
    f"{CA}.artifacts.writers",
    f"{CA}.reporting.metadata",
    f"{CA}.archive.manifest",
    f"{CA}.cli.compare_rendering",
    f"{CA}.cli.output",
    f"{CA}.diagnostic_trace",
    "nlpo_toolkit.latin.cleaners.config_loader",
    "nlpo_toolkit.latin.latin_wordlist.build_latin_wordlist",
)

PACKAGE_CYCLE_GROUPS = (
    "nlpo_toolkit.backends",
    "nlpo_toolkit.comparison",
    "nlpo_toolkit.configuration",
    "nlpo_toolkit.corpus_analysis.analysis_cache",
    "nlpo_toolkit.corpus_analysis.archive",
    "nlpo_toolkit.corpus_analysis.artifacts",
    "nlpo_toolkit.corpus_analysis.cli",
    "nlpo_toolkit.corpus_analysis.features",
    "nlpo_toolkit.corpus_analysis.planning",
    "nlpo_toolkit.corpus_analysis.postprocessing",
    "nlpo_toolkit.corpus_analysis.reporting",
    "nlpo_toolkit.corpus_analysis.token_artifact",
    "nlpo_toolkit.corpus_analysis.token_sequences",
    "nlpo_toolkit.latin.cleaners",
    "nlpo_toolkit.nlp",
    "nlpo_toolkit.serialization",
)

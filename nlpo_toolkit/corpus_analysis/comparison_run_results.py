from __future__ import annotations

from dataclasses import dataclass

from nlpo_toolkit.comparison.config import ComparisonSpec
from nlpo_toolkit.comparison.engine import FrequencyTable
from nlpo_toolkit.comparison.results import ConfiguredComparisonResult


@dataclass(frozen=True)
class ConfiguredComparisonsRunResult:
    comparisons: tuple[
        ConfiguredComparisonResult[ComparisonSpec, FrequencyTable], ...
    ] = ()


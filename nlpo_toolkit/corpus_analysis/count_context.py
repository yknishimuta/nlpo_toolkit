from __future__ import annotations

from dataclasses import dataclass

from .artifacts.models import ArtifactPlan
from .session_models import NLPExecutionSession


@dataclass(frozen=True)
class CountRunContext:
    session: NLPExecutionSession
    artifact_plan: ArtifactPlan


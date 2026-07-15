from types import MappingProxyType

from nlpo_toolkit.cleaner_contracts import CleanerKind

from .corpora.corpus_corporum import PROFILE as CORPUS_CORPORUM_PROFILE
from .corpora.scholastic import PROFILE as SCHOLASTIC_PROFILE
from .errors import UnknownCleanerKindError
from .models import CleanerProfile


_PROFILES = MappingProxyType({profile.kind: profile for profile in (CORPUS_CORPORUM_PROFILE, SCHOLASTIC_PROFILE)})


def get_cleaner_profile(kind: CleanerKind) -> CleanerProfile:
    try:
        return _PROFILES[kind]
    except KeyError as exc:
        raise UnknownCleanerKindError(f"Unknown cleaner kind: {kind!r}") from exc

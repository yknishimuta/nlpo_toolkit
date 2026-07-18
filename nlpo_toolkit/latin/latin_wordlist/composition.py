from .collectors import (
    collect_conllu_candidates,
    collect_extra_wordlist_candidates,
    collect_text_candidates,
)
from .ports import LatinWordlistDependencies
from .publication import publish_wordlist


def default_latin_wordlist_dependencies() -> LatinWordlistDependencies:
    return LatinWordlistDependencies(
        collect_conllu=collect_conllu_candidates,
        collect_text=collect_text_candidates,
        collect_extra_wordlist=collect_extra_wordlist_candidates,
        publish=publish_wordlist,
    )

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from nlpo_toolkit.nlp.contracts import UDMorphFeature

from ..analysis_records import NLPAnalysisRecord
from .errors import FeatureError
from .models import AnalyzedFeatureCorpus, FeatureScalar, MorphologyOptions


def encode_morphology_column_component(value: str) -> str:
    return "".join(
        character
        if character.isascii() and character.isalnum()
        else f"_u{ord(character):06x}_"
        for character in value
    )


@dataclass(frozen=True, order=True)
class MorphologyBundle:
    features: tuple[UDMorphFeature, ...]

    def __post_init__(self) -> None:
        features = tuple(sorted(self.features))
        if not features:
            raise FeatureError("morphology bundle must not be empty")
        if len({item.attribute for item in features}) != len(features):
            raise FeatureError("duplicate attribute in morphology bundle")
        object.__setattr__(self, "features", features)


@dataclass(frozen=True)
class MorphologyVocabulary:
    attributes: tuple[str, ...]
    values: tuple[UDMorphFeature, ...]
    bundles: tuple[MorphologyBundle, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "attributes", tuple(self.attributes))
        object.__setattr__(self, "values", tuple(self.values))
        object.__setattr__(self, "bundles", tuple(self.bundles))
        if not self.attributes or len(self.attributes) != len(set(self.attributes)):
            raise FeatureError("morphology vocabulary attributes must be unique")


def canonical_morphology_bundle(
    morphology: tuple[UDMorphFeature, ...], *, attributes: tuple[str, ...]
) -> MorphologyBundle | None:
    allowed = frozenset(attributes)
    projected = tuple(item for item in morphology if item.attribute in allowed)
    return MorphologyBundle(projected) if projected else None


def _bundle_text(bundle: MorphologyBundle) -> str:
    return "|".join(f"{item.attribute}={item.value}" for item in bundle.features)


def _bundle_column(bundle: MorphologyBundle) -> str:
    encoded = "_p_".join(
        f"{encode_morphology_column_component(item.attribute)}_e_"
        f"{encode_morphology_column_component(item.value)}"
        for item in bundle.features
    )
    return f"morph_bundle_{encoded}"


def fit_morphology_vocabulary(
    corpora: tuple[AnalyzedFeatureCorpus, ...], *, options: MorphologyOptions
) -> MorphologyVocabulary:
    records = tuple(record for corpus in corpora for record in corpus.lexical_records)
    annotated = tuple(record for record in records if record.morphology)
    if not annotated:
        raise FeatureError(
            "morphological features were requested, but the NLP backend produced "
            "no UD FEATS for eligible tokens"
        )
    discovered = {item.attribute for record in annotated for item in record.morphology}
    attributes = options.attributes or tuple(sorted(discovered))
    if options.attributes and not any(item in discovered for item in attributes):
        raise FeatureError(
            "requested morphology attributes were not produced: "
            + ", ".join(attributes)
        )
    values = tuple(
        UDMorphFeature(attribute, value)
        for attribute in attributes
        for value in sorted(
            {
                item.value
                for record in annotated
                for item in record.morphology
                if item.attribute == attribute
            }
        )
    )
    bundles: tuple[MorphologyBundle, ...] = ()
    if options.bundle_top is not None:
        counts = Counter(
            bundle
            for record in annotated
            if (
                bundle := canonical_morphology_bundle(
                    record.morphology, attributes=attributes
                )
            )
            is not None
        )
        if not counts:
            raise FeatureError("no morphology bundles can be generated")
        bundles = tuple(
            bundle
            for bundle, _count in sorted(
                counts.items(), key=lambda item: (-item[1], _bundle_text(item[0]))
            )[: options.bundle_top]
        )
    vocabulary = MorphologyVocabulary(attributes, values, bundles)
    columns = morphology_columns(vocabulary)
    if len(columns) != len(set(columns)):
        raise FeatureError("duplicate morphology feature column")
    return vocabulary


def morphology_columns(vocabulary: MorphologyVocabulary) -> tuple[str, ...]:
    encoded_attributes = tuple(
        (attribute, encode_morphology_column_component(attribute))
        for attribute in vocabulary.attributes
    )
    columns = [
        f"morph_coverage_{encoded}" for _attribute, encoded in encoded_attributes
    ]
    for prefix in ("morph_value", "morph_conditional"):
        columns.extend(
            f"{prefix}_{encode_morphology_column_component(item.attribute)}_"
            f"{encode_morphology_column_component(item.value)}"
            for item in vocabulary.values
        )
    columns.extend(
        f"morph_other_{encoded}" for _attribute, encoded in encoded_attributes
    )
    columns.extend(
        f"morph_other_conditional_{encoded}"
        for _attribute, encoded in encoded_attributes
    )
    if vocabulary.bundles:
        columns.append("morph_bundle_coverage")
        columns.extend(_bundle_column(bundle) for bundle in vocabulary.bundles)
        columns.append("morph_bundle_other")
    return tuple(columns)


def compute_morphology_features(
    records: tuple[NLPAnalysisRecord, ...], *, vocabulary: MorphologyVocabulary
) -> dict[str, FeatureScalar]:
    denominator = len(records)
    rows = tuple(
        {item.attribute: item.value for item in record.morphology} for record in records
    )
    computed: dict[str, FeatureScalar] = {}
    known_by_attribute = {
        attribute: {
            item.value for item in vocabulary.values if item.attribute == attribute
        }
        for attribute in vocabulary.attributes
    }
    for attribute in vocabulary.attributes:
        encoded_attribute = encode_morphology_column_component(attribute)
        annotated = tuple(row[attribute] for row in rows if attribute in row)
        coverage_count = len(annotated)
        computed[f"morph_coverage_{encoded_attribute}"] = (
            coverage_count / denominator if denominator else 0.0
        )
        counts = Counter(annotated)
        for value in sorted(known_by_attribute[attribute]):
            encoded_value = encode_morphology_column_component(value)
            count = counts[value]
            computed[f"morph_value_{encoded_attribute}_{encoded_value}"] = (
                count / denominator if denominator else 0.0
            )
            computed[f"morph_conditional_{encoded_attribute}_{encoded_value}"] = (
                count / coverage_count if coverage_count else 0.0
            )
        other = sum(
            count
            for value, count in counts.items()
            if value not in known_by_attribute[attribute]
        )
        computed[f"morph_other_{encoded_attribute}"] = (
            other / denominator if denominator else 0.0
        )
        computed[f"morph_other_conditional_{encoded_attribute}"] = (
            other / coverage_count if coverage_count else 0.0
        )
    if vocabulary.bundles:
        bundles = tuple(
            bundle
            for record in records
            if (
                bundle := canonical_morphology_bundle(
                    record.morphology, attributes=vocabulary.attributes
                )
            )
            is not None
        )
        counts = Counter(bundles)
        selected = frozenset(vocabulary.bundles)
        computed["morph_bundle_coverage"] = (
            len(bundles) / denominator if denominator else 0.0
        )
        for bundle in vocabulary.bundles:
            computed[_bundle_column(bundle)] = (
                counts[bundle] / denominator if denominator else 0.0
            )
        computed["morph_bundle_other"] = (
            sum(count for bundle, count in counts.items() if bundle not in selected)
            / denominator
            if denominator
            else 0.0
        )
    return {column: computed[column] for column in morphology_columns(vocabulary)}

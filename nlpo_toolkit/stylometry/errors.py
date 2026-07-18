class StylometryError(ValueError):
    """Base error for stylometry input and calculation failures."""


class StylometryMetricError(StylometryError):
    """Raised when a vector metric cannot be calculated."""

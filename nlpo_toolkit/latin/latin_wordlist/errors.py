class LatinWordlistError(Exception):
    """Base error for the Latin wordlist builder."""


class LatinWordlistConfigError(LatinWordlistError):
    """The wordlist configuration is invalid or unreadable."""


class LatinWordlistSourceReadError(LatinWordlistError):
    """A configured source could not be read as strict UTF-8."""


class LatinWordlistSourceParseError(LatinWordlistError):
    """A configured source contains invalid structured data."""


class LatinWordlistPublicationError(LatinWordlistError):
    """The completed wordlist could not be published atomically."""

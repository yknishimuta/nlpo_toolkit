class CleanerError(ValueError):
    pass


class CleanerRuleConfigError(CleanerError):
    pass


class CleanerLexiconError(CleanerError):
    pass


class UnknownCleanerKindError(CleanerError):
    pass

class ComparisonError(ValueError):
    pass


class ComparisonEngineError(ComparisonError):
    pass


class ComparisonInputError(ComparisonError):
    pass


class FrequencyTableReadError(ComparisonInputError):
    pass


class ComparisonServiceError(ComparisonError):
    pass

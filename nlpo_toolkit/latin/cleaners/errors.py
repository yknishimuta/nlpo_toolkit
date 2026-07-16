from nlpo_toolkit.cleaner_contracts import CleanerApplicationError


class CleanerError(CleanerApplicationError, ValueError):
    pass


class CleanerRuleConfigError(CleanerError):
    pass


class CleanerLexiconError(CleanerError):
    pass


class UnknownCleanerKindError(CleanerError):
    pass


class CleanerExecutionError(CleanerApplicationError, RuntimeError):
    pass


class CleanerInputReadError(CleanerExecutionError):
    pass


class CleanerOutputPlanError(CleanerExecutionError):
    pass


class CleanerTemplateError(CleanerOutputPlanError):
    pass


class CleanerOutputWriteError(CleanerExecutionError):
    pass

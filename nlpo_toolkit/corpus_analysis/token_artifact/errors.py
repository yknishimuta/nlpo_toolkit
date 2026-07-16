class TokenArtifactError(ValueError):
    pass


class TokenArtifactMetadataError(TokenArtifactError):
    pass


class TokenArtifactFormatError(TokenArtifactError):
    pass


class TokenArtifactRowError(TokenArtifactFormatError):
    pass


class TokenArtifactIntegrityError(TokenArtifactError):
    pass


class TokenArtifactWriterStateError(TokenArtifactError):
    pass

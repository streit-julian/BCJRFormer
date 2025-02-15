from enum import IntEnum, StrEnum


# Booleans don't work so well with Spock CLI, so we do this "hack" to make them work
class BooleanFlag(IntEnum):
    FALSE = 0
    TRUE = 1


class SchedulerType(StrEnum):
    COSINE = "cosine"
    CONSTANT = "constant"

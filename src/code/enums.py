from enum import Enum


class ChunkType(Enum):
    METHOD = "method"
    CLASS = "class"
    FUNCTION = "function"
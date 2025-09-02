from dataclasses import dataclass
from typing import Optional
from .enums import ChunkType

@dataclass
class CodeChunk:
    content: str
    chunk_type: ChunkType
    name: str
    file_path: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parent_class: Optional[str] = None
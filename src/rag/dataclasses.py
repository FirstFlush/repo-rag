from dataclasses import dataclass
from typing import Any, Optional
from src.chat.dataclasses import ChatResponse


@dataclass
class RAGResponse:
    
    sources: list[dict[str, Any]]
    question: str
    num_chunks_retrieved: int
    chat_response: Optional[ChatResponse] = None
    
    @property
    def answer(self) -> str:
        """Convenience property for just the text answer."""
        return self.chat_response.content if self.chat_response else ""
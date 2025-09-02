from dataclasses import dataclass
from typing import Any, Optional
from .enums import ChatRole, ChatClientEnum
from src.config.constants import ModelConstants

@dataclass
class ChatMessage:
    """Represents a single message in a conversation."""
    role: ChatRole
    content: str

@dataclass
class ChatResponse:
    """Standardized response from any chat API."""
    content: str
    model: str
    input_tokens: int
    output_tokens: int
    finish_reason: str
    client: ChatClientEnum
    request_id: str
    raw_response: Optional[Any] = None  # Store original response for debugging
    latency_ms: Optional[int] = None # Optional for added context

@dataclass
class ChatInput:
    messages: list[ChatMessage]
    model: Optional[str] = None # None to use client's default
    temperature: float = ModelConstants.TEMPERATURE
    max_tokens: int = ModelConstants.MAX_TOKENS
    
    @staticmethod
    def build_messages(messages: list[dict[str, Any]]) -> list[ChatMessage]:
        message_objects: list[ChatMessage] = []
        for message in messages:
            message_objects.append(
                ChatMessage(role=message["role"], content=message["content"])
            )
        return message_objects
    
    @property
    def messages_normalized(self) -> list[dict[str, str]]:
        return [
            {
                "role": message.role.value,
                "content": message.content,
            } for message in self.messages
        ]
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Type
from .dataclasses import ChatMessage, ChatResponse, ChatInput
from .enums import ChatRole
from .exc import ChatClientError
from src.config.prompts import UserPrompt, SystemPrompt
from src.config.constants import ModelConstants
from src.config.logging import get_logger

logger = get_logger(__name__)

class BaseChatClient(ABC):
    """Base class for all LLM API clients."""

    @property
    @abstractmethod
    def models(self) -> Type[Enum]:
        """Available models for this provider."""
        pass
    
    @property 
    @abstractmethod
    def default_model(self) -> Enum:
        """Default model for this provider."""
        pass
    
    def __init__(self, api_key: str):
        self.api_key = api_key

    def chat(self, chat_input: ChatInput) -> ChatResponse:
        try:
            response = self._chat(chat_input)
        except Exception as e:
            msg = f"{self.__class__.__name__} API call failed due to an unexpected error: {e}"
            logger.error(msg, exc_info=True)
            raise ChatClientError(msg) from e
        else:
            return response

    @abstractmethod
    def _chat(self, chat_input: ChatInput) -> ChatResponse:
        """Generate a chat completion."""
        pass

    def list_models(self) -> list[str]:
        """Return list of available models for this provider."""
        return [model.value for model in self.models]
    
    def chat_simple(
            self, 
            system_prompt: str, 
            user_prompt: str,
            model: Optional[str] = None,
            temperature: float = ModelConstants.TEMPERATURE,
            max_tokens: int = ModelConstants.MAX_TOKENS
    ) -> str:
        """
        Convenience method for simple system + user prompt.
        Returns just the content string.
        """
        messages = [
            ChatMessage(role=ChatRole.SYSTEM, content=system_prompt),
            ChatMessage(role=ChatRole.USER, content=user_prompt)
        ]
        
        chat_input = ChatInput(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens
        )
        response = self.chat(chat_input)
        
        return response.content
    
    def validate_config(self) -> bool:
        """
        Validate that the client is properly configured.
        Should check API key, connectivity, etc.
        """
        try:
            # Try a minimal API call
            test_response = self.chat_simple(
                system_prompt=SystemPrompt.TEST,
                user_prompt=UserPrompt.TEST,
                max_tokens=10
            )
            return "OK" in test_response.upper()
        except Exception:
            return False
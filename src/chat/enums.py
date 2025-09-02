from enum import Enum


class ChatClientEnum(Enum):
    OPENAI = "openai"

class ChatRole(Enum):
    
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
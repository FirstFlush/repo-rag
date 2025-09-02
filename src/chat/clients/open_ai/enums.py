from enum import Enum

class OpenAiModels(Enum):
    """Current OpenAI API models as of September 2025."""
    
    # GPT-5 Series (Latest flagship)
    GPT_5 = "gpt-5"
    GPT_5_MINI = "gpt-5-mini"
    GPT_5_NANO = "gpt-5-nano"
    GPT_5_CHAT_LATEST = "gpt-5-chat-latest"
    
    # GPT-4.1 Series (API-focused, high performance)
    GPT_4_1 = "gpt-4.1"
    GPT_4_1_MINI = "gpt-4.1-mini"
    GPT_4_1_NANO = "gpt-4.1-nano"
    
    # GPT-4o Series (Multimodal)
    GPT_4O = "gpt-4o"
    GPT_4O_2024_05_13 = "gpt-4o-2024-05-13"
    GPT_4O_MINI = "gpt-4o-mini"
    
    # Reasoning Models (o-series)
    O3 = "o3"
    O3_MINI = "o3-mini"
    O3_PRO = "o3-pro"
    O4_MINI = "o4-mini"
    O4_MINI_HIGH = "o4-mini-high"
    
    # Legacy GPT-4 (Still available)
    GPT_4 = "gpt-4"
    GPT_4_TURBO = "gpt-4-turbo"
    
    # Specialized Models
    GPT_4O_AUDIO_PREVIEW = "gpt-4o-audio-preview"
    GPT_4O_MINI_AUDIO_PREVIEW = "gpt-4o-mini-audio-preview"
    GPT_4O_REALTIME_PREVIEW = "gpt-4o-realtime-preview"
    
    # Image Generation
    GPT_IMAGE_1 = "gpt-image-1"
    
    # Open Source Models
    GPT_OSS_120B = "gpt-oss-120b"
    GPT_OSS_20B = "gpt-oss-20b"
    
    # Legacy (may be deprecated soon)
    GPT_3_5_TURBO = "gpt-3.5-turbo"

# Usage examples:
# model = OpenAiModels.GPT_4O.value  # "gpt-4o"
# default_model = OpenAiModels.GPT_4O_MINI  # For cost efficien
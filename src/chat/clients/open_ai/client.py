from ...base import BaseChatClient
from ...dataclasses import ChatInput, ChatResponse, ChatMessage
from ...enums import ChatRole, ChatClientEnum
from .enums import OpenAiModels
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletion
from typing import Literal, Type
from src.config.logging import get_logger

logger = get_logger(__name__)

class OpenAiClient(BaseChatClient):

    enum = ChatClientEnum.OPENAI
    
    def __init__(self, api_key: str):
        self._client = OpenAI(api_key=api_key)
    
    def _format_messages(self, chat_input: ChatInput) -> list[ChatCompletionMessageParam]:
        msgs: list[ChatCompletionMessageParam] = chat_input.messages_normalized # type: ignore
        return msgs
    
    def _chat(self, chat_input: ChatInput) -> ChatResponse:
        model = chat_input.model if chat_input.model else self.default_model.value
        try:
            chat_completion = self._client.chat.completions.create(
                model=model,
                messages=self._format_messages(chat_input),
                temperature=chat_input.temperature,
                max_tokens=chat_input.max_tokens,
            )
        
        except Exception as e:
            logger.error(f"Error generating answer: {str(e)}", exc_info=True)
            raise
    
        else:
            return self._build_response(chat_completion)
        
    def _content(self, completion: ChatCompletion) -> str | None:
        return completion.choices[0].message.content
        
    def _build_response(self, completion: ChatCompletion) -> ChatResponse:
        usage = completion.usage
        if usage:
            input_tokens = usage.prompt_tokens
            output_tokens = usage.completion_tokens
        else:
            input_tokens = 0
            output_tokens = 0
        
        return ChatResponse(
            model=completion.model,
            content=self._content(completion) or "",
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=completion.choices[0].finish_reason,
            client=ChatClientEnum.OPENAI,
            request_id=completion.id
        )
        


    # def _generate_answer(self, question: str, context: str) -> str:
    #     """Generate answer using OpenAI API."""
    #     system_prompt = SystemPrompt.SYSTEM_PROMPT
    #     user_prompt = UserPrompt.build_user_prompt(question=question, context=context)

    #     try:
    #         response = self.client.chat.completions.create(
    #             model=self.model,
    #             messages=[
    #                 {"role": "system", "content": system_prompt},
    #                 {"role": "user", "content": user_prompt}
    #             ],
    #             temperature=ModelConstants.TEMPERATURE,
    #             max_tokens=ModelConstants.MAX_TOKENS,
    #         )
        
    #     except Exception as e:
    #         return f"Error generating answer: {str(e)}"
    
    #     else:
    #         content = response.choices[0].message.content
    #         if content is None:
    #             raise ValueError("Response content is empty") 
    #         return content.strip()
            


    @property
    def models(self) -> Type[OpenAiModels]:
        return OpenAiModels

    @property
    def default_model(self) -> Literal[OpenAiModels.GPT_4_1_MINI]:
        return OpenAiModels.GPT_4_1_MINI
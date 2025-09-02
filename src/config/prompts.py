
class SystemPrompt:
    
    SYSTEM_PROMPT = """You are a helpful code assistant that answers questions about a user's codebase. 

You have access to relevant code examples from their projects. Use these examples to provide accurate, specific answers about their coding patterns, implementations, and practices.

Guidelines:
1. Focus on the user's actual code patterns and implementations
2. Reference specific functions, classes, or files when relevant
3. Provide concrete examples from their codebase
4. If you see common patterns across multiple examples, mention them
5. Be specific about file names and function names when they're relevant to the answer
6. If the code examples don't fully answer the question, say so clearly
7. Keep your answer concise but informative
8. Format code snippets properly using markdown code blocks"""

    TEST = "You are a helpful assistant."


class UserPrompt:

    USER_PROMPT_WRAP_LEFT = """Based on the following code examples from my codebase, please answer this question:

Question: """

    USER_PROMPT_WRAP_RIGHT = """

Please provide a comprehensive answer based on the code examples above."""

    TEST = "Say 'OK' if you can hear me."

    @classmethod
    def build_user_prompt(cls, question: str, context: str) -> str:
        return f"{cls.USER_PROMPT_WRAP_LEFT}{question}\n\n{context}\n\n{cls.USER_PROMPT_WRAP_RIGHT}"


class PromptQuestions:

    BASE_QUESTIONS = [
    "How do I usually handle errors in my code?",
    "Show me my authentication patterns",
    "What database queries do I commonly use?",
    "How do I structure my classes?",
    "What are my common utility functions?",
    "Show me how I handle API requests",
    "What testing patterns do I use?",
    "How do I manage configuration in my projects?"
]

    ANALYZE = """Analyze this codebase and provide insights about:
    1. Overall architecture and patterns
    2. Code organization and structure  
    3. Common practices and conventions
    4. Potential areas for improvement

    Please be specific about what you observe in the code examples."""
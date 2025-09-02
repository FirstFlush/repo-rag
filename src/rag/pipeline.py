
from typing import List, Dict, Any, Tuple
from src.config.prompts import SystemPrompt, UserPrompt, PromptQuestions
from src.code.parser import CodeChunk
from src.config.constants import ModelConstants
from src.chat.base import BaseChatClient
from src.chat.dataclasses import ChatInput, ChatResponse
from src.chat.enums import ChatRole
from src.rag.embeddings import EmbeddingSystem
from .dataclasses import RAGResponse

class RAGPipeline:
    def __init__(
            self, 
            embedding_system: EmbeddingSystem,
            chat_client: BaseChatClient,
    ):
        """Initialize RAG pipeline with embedding system and OpenAI client."""
        self.embedding_system = embedding_system
        self.client = chat_client
        
        # Default model for completions
        self.model = ModelConstants.MODEL
    
    def ask(self, question: str, max_chunks: int = 5, include_context: bool = True) -> RAGResponse:
        """Query the codebase using RAG pipeline."""
        
        # Step 1: Retrieve relevant code chunks
        relevant_chunks = self.embedding_system.search(question, k=max_chunks)
        if not relevant_chunks:
            return RAGResponse(
                chat_response=None,
                sources=[],
                question=question,
                num_chunks_retrieved=0,
            )
        context = self._build_context(relevant_chunks, include_metadata=include_context)
        
        chat_input = self._chat_input(question=question, context=context)
        response = self._generate_response(chat_input)
        
        # Step 4: Format response
        sources = self._format_sources(relevant_chunks)
        
        return RAGResponse(
            chat_response=response,
            sources=sources,
            question=question,
            num_chunks_retrieved=len(relevant_chunks),
        )
        

    def _chat_input(self, question: str, context: str) -> ChatInput:
        system_prompt = SystemPrompt.SYSTEM_PROMPT
        user_prompt = UserPrompt.build_user_prompt(question=question, context=context)
        messages_raw = [
            {"role": ChatRole.SYSTEM, "content": system_prompt},
            {"role": ChatRole.USER, "content": user_prompt},
        ]
        return ChatInput(messages=ChatInput.build_messages(messages_raw))


    def _generate_response(self, chat_input: ChatInput) -> ChatResponse:
        return self.client.chat(chat_input)
        

    def _build_context(self, chunks_with_scores: List[Tuple[CodeChunk, float]], 
                      include_metadata: bool = True) -> str:
        """Build context string from retrieved code chunks."""
        context_parts = ["Here are relevant code examples from your codebase:\n"]
        
        for i, (chunk, score) in enumerate(chunks_with_scores, 1):
            context_parts.append(f"=== Code Example {i} ===")
            
            if include_metadata:
                # Add metadata
                metadata = []
                metadata.append(f"File: {chunk.file_path}")
                metadata.append(f"Type: {chunk.chunk_type}")
                metadata.append(f"Name: {chunk.name}")
                if chunk.parent_class:
                    metadata.append(f"Class: {chunk.parent_class}")
                metadata.append(f"Lines: {chunk.start_line}-{chunk.end_line}")
                metadata.append(f"Relevance Score: {score:.3f}")
                
                context_parts.append("Metadata: " + " | ".join(metadata))
                
                if chunk.docstring:
                    context_parts.append(f"Documentation: {chunk.docstring}")
                
                context_parts.append("Code:")
            
            # Add the actual code
            context_parts.append(chunk.content)
            context_parts.append("")  # Empty line between chunks
        
        return "\n".join(context_parts)
    
            
    def _format_sources(self, chunks_with_scores: List[Tuple[CodeChunk, float]]) -> List[Dict[str, Any]]:
        """Format source information for display."""
        sources = []
        
        for chunk, score in chunks_with_scores:
            source = {
                'file_path': chunk.file_path,
                'chunk_type': chunk.chunk_type,
                'name': chunk.name,
                'start_line': chunk.start_line,
                'end_line': chunk.end_line,
                'relevance_score': round(score, 3),
                'parent_class': chunk.parent_class
            }
            sources.append(source)
        
        return sources
    
    def suggest_questions(self) -> List[str]:
        """Suggest example questions based on the codebase."""
        stats = self.embedding_system.get_stats()
        
        base_questions = PromptQuestions.BASE_QUESTIONS
        
        # Add language-specific questions
        languages = stats.get('languages', [])
        if '.py' in languages:
            base_questions.extend([
                "How do I handle decorators in Python?",
                "Show me my Python class inheritance patterns"
            ])
        
        if any(ext in languages for ext in ['.js', '.ts', '.jsx', '.tsx']):
            base_questions.extend([
                "What are my React component patterns?",
                "How do I handle async operations in JavaScript?"
            ])
        
        return base_questions
    
    def analyze_codebase(self) -> Dict[str, Any]:
        """Provide a general analysis of the codebase."""
        stats = self.embedding_system.get_stats()
        
        question = PromptQuestions.ANALYZE
        
        # Get a broader sample of chunks
        sample_chunks = self.embedding_system.search("architecture patterns structure", k=10)
        context = self._build_context(sample_chunks, include_metadata=True)
        chat_input = self._chat_input(question=question, context=context)
        analysis = self._generate_response(chat_input)
        
        return {
            'analysis': analysis,
            'stats': stats,
            'sample_chunks': len(sample_chunks)
        }

# if __name__ == "__main__":
#     # Test the RAG pipeline
#     from src.rag.embeddings import EmbeddingSystem
#     import os
    
#     if not os.getenv('OPENAI_API_KEY'):
#         print("Please set OPENAI_API_KEY environment variable to test the RAG pipeline")
#         exit(1)
    
#     # Initialize systems
#     embedding_system = EmbeddingSystem()
#     rag = RAGPipeline(embedding_system)
    
#     # Test query
#     result = rag.query("How do I save data to disk?")
#     print("Answer:", result['answer'])
#     print("\nSources:")
#     for source in result['sources']:
#         print(f"  - {source['name']} in {source['file_path']}")
        
        
        
        
        
        
        
        

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
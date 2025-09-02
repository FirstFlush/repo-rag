import streamlit as st
import os
from pathlib import Path

from src.chat.clients.open_ai.client import OpenAiClient
from src.code.scanner import CodeScanner
from src.code.parser import CodeParser
from src.rag.embeddings import EmbeddingSystem
from src.rag.pipeline import RAGPipeline

# Page configuration
st.set_page_config(
    page_title="Codebase RAG System",
    page_icon="🔍",
    layout="wide"
)

# Initialize session state
if 'embedding_system' not in st.session_state:
    st.session_state.embedding_system = None
if 'rag_pipeline' not in st.session_state:
    st.session_state.rag_pipeline = None
if 'indexed_directories' not in st.session_state:
    st.session_state.indexed_directories = []

def initialize_systems():
    """Initialize the embedding and RAG systems."""
    if st.session_state.embedding_system is None:
        with st.spinner("Initializing embedding system..."):
            st.session_state.embedding_system = EmbeddingSystem()
    
    # Only initialize RAG pipeline if OpenAI API key is available
    openai_key = st.session_state.get('openai_api_key') or os.getenv('OPENAI_API_KEY', "")
    client = OpenAiClient(openai_key)
    if openai_key and st.session_state.rag_pipeline is None:
        try:
            st.session_state.rag_pipeline = RAGPipeline(
                st.session_state.embedding_system, 
                client,
            )
        except Exception as e:
            st.error(f"Failed to initialize RAG pipeline: {e}")

def scan_and_index_directory(directory: str):
    """Scan a directory and index all code files."""
    if not directory or not Path(directory).exists():
        st.error("Directory does not exist!")
        return False
    
    try:
        with st.spinner(f"Scanning directory: {directory}"):
            scanner = CodeScanner()
            files = scanner.scan_directory(directory)
            
            if not files:
                st.warning("No supported code files found!")
                return False
            
            st.info(f"Found {len(files)} code files")
        
        with st.spinner("Parsing code files..."):
            parser = CodeParser()
            all_chunks = []
            
            progress_bar = st.progress(0)
            for i, file_path in enumerate(files):
                chunks = parser.parse_file(file_path)
                all_chunks.extend(chunks)
                progress_bar.progress((i + 1) / len(files))
            
            st.info(f"Extracted {len(all_chunks)} code chunks")
        
        with st.spinner("Computing embeddings and building search index..."):
            st.session_state.embedding_system.add_chunks(all_chunks)
        
        # Add to indexed directories
        abs_directory = str(Path(directory).resolve())
        if abs_directory not in st.session_state.indexed_directories:
            st.session_state.indexed_directories.append(abs_directory)
        
        st.success(f"Successfully indexed {len(all_chunks)} code chunks from {directory}")
        return True
        
    except Exception as e:
        st.error(f"Error indexing directory: {e}")
        return False

def main():
    st.title("🔍 Codebase RAG System")
    st.markdown("Search and ask questions about your code across projects")
    
    # Initialize systems
    initialize_systems()
    
    # Sidebar for configuration and indexing
    with st.sidebar:
        st.header("Configuration")
        
        # OpenAI API Key
        api_key_input = st.text_input(
            "OpenAI API Key", 
            type="password",
            value=st.session_state.get('openai_api_key', ''),
            help="Required for generating answers. Can also be set as OPENAI_API_KEY environment variable."
        )
        
        if api_key_input:
            st.session_state.openai_api_key = api_key_input
            # Reinitialize RAG pipeline if key changed
            if st.session_state.rag_pipeline is None:
                initialize_systems()
        
        st.divider()
        
        # Directory indexing
        st.header("Index Codebase")
        
        directory_input = st.text_input(
            "Directory to scan",
            placeholder="/path/to/your/project",
            help="Path to directory containing your code files"
        )
        
        if st.button("Index Directory", type="primary"):
            if directory_input:
                scan_and_index_directory(directory_input)
            else:
                st.error("Please enter a directory path")
        
        # Show current status
        if st.session_state.embedding_system:
            stats = st.session_state.embedding_system.get_stats()
            
            st.divider()
            st.header("Current Status")
            st.metric("Total Code Chunks", stats.get('total_chunks', 0))
            st.metric("Total Files", stats.get('total_files', 0))
            
            if stats.get('chunk_types'):
                st.write("**Chunk Types:**")
                for chunk_type, count in stats['chunk_types'].items():
                    st.write(f"- {chunk_type}: {count}")
            
            if stats.get('languages'):
                st.write("**Languages:**")
                st.write(", ".join(stats['languages']))
        
        # Clear data option
        st.divider()
        if st.button("Clear All Data", type="secondary"):
            if st.session_state.embedding_system:
                st.session_state.embedding_system.clear_data()
                st.session_state.indexed_directories = []
                st.success("All data cleared!")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("Ask Questions About Your Code")
        
        # Check if system is ready
        if not st.session_state.embedding_system or st.session_state.embedding_system.get_stats()['total_chunks'] == 0:
            st.warning("👈 Please index some code directories first using the sidebar")
            return
        
        if not st.session_state.rag_pipeline:
            st.warning("👈 Please enter your OpenAI API key in the sidebar to enable Q&A")
            return
        
        # Query input
        query = st.text_input(
            "Your question:",
            placeholder="How do I usually handle authentication?",
            help="Ask questions about your coding patterns, implementations, or specific functions"
        )
        
        # Search parameters
        with st.expander("Search Settings"):
            max_chunks = st.slider("Max code examples to retrieve", 1, 10, 5)
            include_context = st.checkbox("Include metadata in context", True)
        
        if st.button("Search & Answer", type="primary") and query:
            with st.spinner("Searching codebase and generating answer..."):
                try:
                    result = st.session_state.rag_pipeline.query(
                        query, 
                        max_chunks=max_chunks,
                        include_context=include_context
                    )
                    
                    # Display answer
                    st.subheader("Answer")
                    st.markdown(result['answer'])
                    
                    # Display sources
                    if result['sources']:
                        st.subheader("Sources")
                        for i, source in enumerate(result['sources'], 1):
                            with st.expander(f"Source {i}: {source['name']} ({source['chunk_type']})"):
                                st.write(f"**File:** {source['file_path']}")
                                st.write(f"**Lines:** {source['start_line']}-{source['end_line']}")
                                st.write(f"**Relevance:** {source['relevance_score']}")
                                if source['parent_class']:
                                    st.write(f"**Class:** {source['parent_class']}")
                
                except Exception as e:
                    st.error(f"Error processing query: {e}")
    
    with col2:
        st.header("Suggested Questions")
        
        if st.session_state.rag_pipeline:
            suggestions = st.session_state.rag_pipeline.suggest_questions()
            
            for suggestion in suggestions[:8]:  # Show first 8 suggestions
                if st.button(suggestion, key=f"suggestion_{suggestion}"):
                    st.session_state.temp_query = suggestion
                    st.rerun()
        
        # Handle suggestion clicks
        if hasattr(st.session_state, 'temp_query'):
            query = st.session_state.temp_query
            del st.session_state.temp_query
        
        # Codebase Analysis
        st.divider()
        st.header("Codebase Analysis")
        
        if st.button("Analyze My Codebase"):
            if st.session_state.rag_pipeline:
                with st.spinner("Analyzing codebase..."):
                    try:
                        analysis = st.session_state.rag_pipeline.analyze_codebase()
                        
                        st.subheader("Analysis Results")
                        st.markdown(analysis['analysis'])
                        
                        with st.expander("Detailed Statistics"):
                            st.json(analysis['stats'])
                    
                    except Exception as e:
                        st.error(f"Error analyzing codebase: {e}")
    
    # Footer
    st.divider()
    st.markdown(
        """
        **How to use:**
        1. Enter your OpenAI API key in the sidebar
        2. Add directories containing your code using the sidebar
        3. Ask questions about your code patterns and implementations
        4. Use suggested questions or analyze your entire codebase
        
        **Supported file types:** .py, .js, .ts, .jsx, .tsx
        """
    )

if __name__ == "__main__":
    main()
# Codebase RAG System

A Retrieval-Augmented Generation (RAG) system that lets you search and ask questions about your own code across projects.

## Features

- =Á **Smart Code Scanning**: Automatically finds `.py`, `.js`, `.ts`, `.jsx`, `.tsx` files while skipping common ignore patterns
- >é **Function-Level Chunking**: Extracts functions, classes, and methods (not arbitrary line splits)
- = **Semantic Search**: Uses sentence-transformers for local embeddings (no API calls for search)
- =¾ **Local Storage**: Everything stored locally using FAISS and JSON (no external databases)
- <¯ **AI-Powered Q&A**: Uses OpenAI API to answer questions based on retrieved code chunks
- =¥ **Streamlit Interface**: Easy-to-use web interface for queries and codebase management

## Installation

1. Clone or download this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Then:
1. Enter your OpenAI API key in the sidebar (or set as environment variable)
2. Add directories containing your code projects
3. Wait for indexing to complete
4. Ask questions about your code!

### Example Questions

- "How do I usually handle authentication?"
- "Show me my error handling patterns"
- "What are my common utility functions?"
- "How do I structure my React components?"
- "What database queries do I use most?"

### Command Line Usage

You can also use the components programmatically:

```python
from code_scanner import CodeScanner
from code_parser import CodeParser
from embeddings import EmbeddingSystem
from rag_pipeline import RAGPipeline

# Scan and parse code
scanner = CodeScanner()
parser = CodeParser()
files = scanner.scan_directory("/path/to/your/project")

all_chunks = []
for file_path in files:
    chunks = parser.parse_file(file_path)
    all_chunks.extend(chunks)

# Create embeddings and search index
embedding_system = EmbeddingSystem()
embedding_system.add_chunks(all_chunks)

# Query your code
rag = RAGPipeline(embedding_system, api_key="your-openai-key")
result = rag.query("How do I handle errors?")
print(result['answer'])
```

## Architecture

### Components

1. **CodeScanner** (`code_scanner.py`): Finds relevant code files while respecting ignore patterns
2. **CodeParser** (`code_parser.py`): Extracts functions, classes, and methods from code files
3. **EmbeddingSystem** (`embeddings.py`): Creates and manages embeddings using sentence-transformers
4. **RAGPipeline** (`rag_pipeline.py`): Combines retrieval and generation for Q&A
5. **StreamlitApp** (`streamlit_app.py`): Web interface for the system

### Data Storage

- **Code chunks**: Stored as JSON with metadata (file path, line numbers, etc.)
- **Embeddings**: Stored as NumPy arrays
- **Search index**: FAISS index for fast similarity search
- **Storage location**: `./embeddings_storage/` directory

### Supported Languages

- Python (`.py`) - Full AST parsing
- JavaScript (`.js`, `.jsx`) - Regex-based parsing
- TypeScript (`.ts`, `.tsx`) - Regex-based parsing

## Configuration

### Ignore Patterns

The scanner automatically ignores common directories and files:
- `node_modules`, `.git`, `__pycache__`
- Virtual environments (`venv`, `.venv`, `env`, `.env`)
- Build directories (`build`, `dist`, `.next`, `.nuxt`)
- Test/cache directories (`.pytest_cache`, `.mypy_cache`)
- Minified files (`*.min.js`, `*.bundle.js`)

### Embedding Model

Default: `all-MiniLM-L6-v2` (384 dimensions, good balance of speed/quality)

You can change this by modifying the `EmbeddingSystem` initialization:

```python
embedding_system = EmbeddingSystem(model_name='all-mpnet-base-v2')  # Higher quality, slower
```

## Troubleshooting

### Common Issues

1. **"No supported code files found"**: Check that your directory contains `.py`, `.js`, `.ts`, `.jsx`, or `.tsx` files
2. **OpenAI API errors**: Ensure your API key is valid and you have sufficient credits
3. **Memory issues**: Large codebases may require more RAM. Consider processing directories separately
4. **Parsing errors**: Some complex JavaScript/TypeScript may not parse perfectly (Python parsing is more robust)

### Performance Tips

- Start with smaller directories to test the system
- Use specific directories rather than entire repositories
- The first run will be slower due to model downloads and embeddings computation
- Subsequent searches are much faster thanks to FAISS indexing

## Limitations

- JavaScript/TypeScript parsing uses regex patterns (less robust than AST)
- Requires OpenAI API key for question answering (embedding generation is local)
- Large codebases may take significant time to index initially
- Context window limits may truncate very long code examples

## Future Enhancements

- Better JavaScript/TypeScript parsing with Tree-sitter
- Support for more languages (Go, Rust, Java, C++)
- Code similarity detection and duplicate finding
- Integration with version control systems
- Codebase evolution tracking over time
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from src.code.parser import CodeChunk
from src.config.logging import get_logger

logger = get_logger(__name__)


class EmbeddingSystem:
    """
    A semantic search system for code chunks using sentence transformers and FAISS.
    
    This class provides functionality to embed code chunks into vector representations
    and perform fast similarity searches using Facebook's FAISS library. It's designed
    for RAG (Retrieval-Augmented Generation) applications where you need to find
    semantically similar code based on natural language queries.
    
    The system combines code content with metadata (function names, docstrings, file paths)
    to create rich text representations that are then embedded using sentence transformers.
    These embeddings are stored in a FAISS index for efficient cosine similarity searches.
    
    Data persistence is handled through JSON serialization for code chunks and numpy
    arrays for embeddings, with automatic loading and saving to disk.
    
    FAISS Integration:
        This implementation uses FAISS (Facebook AI Similarity Search) for fast vector
        similarity search. The system is designed to work with both the high-level Python
        API and the lower-level SWIG bindings, depending on your FAISS installation.
        
        - High-level API: index.add(embeddings), index.search(query, k)
        - SWIG bindings: index.add(n, embeddings), index.search(n, query, k, distances, labels)
        
        The SWIG interface requires manual memory allocation for search results but is
        sometimes the only available interface in certain FAISS installations. This class
        handles the complexity of the SWIG interface internally while providing a clean
        Python API.
        
        Cosine similarity is achieved using IndexFlatL2 with L2-normalized embeddings,
        where cosine_similarity = 1 - (L2_distance² / 2) for normalized vectors.
    """

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', storage_dir: str = './embeddings_storage'):
        self.model_name = model_name
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        logger.info(f"Loading sentence transformer model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # File paths
        self.chunks_file = self.storage_dir / 'code_chunks.json'
        self.embeddings_file = self.storage_dir / 'embeddings.npy'
        self.index_file = self.storage_dir / 'faiss_index.index'
        
        # Data storage
        self.chunks: List[CodeChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.chunk_hashes: set[str] = set()  # Track duplicates
        
        self._load_existing_data()
    
    def _load_existing_data(self):
        """Load existing chunks and embeddings from storage."""
        try:
            if self.chunks_file.exists():
                with open(self.chunks_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                    self.chunks = [self._dict_to_chunk(chunk_dict) for chunk_dict in chunks_data]
                self._update_chunk_hashes()
                logger.info(f"Loaded {len(self.chunks)} existing code chunks")
            
            if self.embeddings_file.exists():
                self.embeddings = np.load(self.embeddings_file)
                if self.embeddings:
                    logger.info(f"Loaded embeddings with shape: {self.embeddings.shape}")
                else:
                    logger.warning("No embeddings found!")
            
            if (self.index_file.exists() and 
                self.embeddings is not None and 
                len(self.chunks) == len(self.embeddings)):
                self.index = faiss.read_index(str(self.index_file))
                logger.info("Loaded FAISS index")
            elif self.embeddings is not None:
                logger.info("Rebuilding FAISS index due to mismatch")
                self._build_faiss_index()
                
        except Exception as e:
            logger.error(f"Error loading existing data: {e}", exc_info=True)
            self._reset_data()
    
    def _reset_data(self):
        """Reset all data structures."""
        self.chunks = []
        self.embeddings = None
        self.index = None
        self.chunk_hashes = set()
    
    def _update_chunk_hashes(self):
        """Update chunk hashes for duplicate detection."""
        self.chunk_hashes = {self._chunk_hash(chunk) for chunk in self.chunks}
    
    def _chunk_hash(self, chunk: CodeChunk) -> str:
        """Create hash for chunk deduplication."""
        return f"{chunk.file_path}:{chunk.start_line}:{chunk.end_line}:{chunk.name}"
    
    def _chunk_to_dict(self, chunk: CodeChunk) -> dict[str, Any]:
        """Convert CodeChunk to dictionary for JSON serialization."""
        return {
            'content': chunk.content,
            'chunk_type': chunk.chunk_type.value,
            'name': chunk.name,
            'file_path': chunk.file_path,
            'start_line': chunk.start_line,
            'end_line': chunk.end_line,
            'docstring': chunk.docstring,
            'parent_class': chunk.parent_class
        }
    
    def _dict_to_chunk(self, chunk_dict: dict[str, Any]) -> CodeChunk:
        """Convert dictionary back to CodeChunk."""
        return CodeChunk(**chunk_dict)
    
    def _create_chunk_text(self, chunk: CodeChunk) -> str:
        """Create searchable text representation of a code chunk."""
        parts = []
        
        # Add metadata for better semantic matching
        if chunk.parent_class:
            parts.append(f"Class: {chunk.parent_class}")
        
        parts.extend([
            f"Type: {chunk.chunk_type}",
            f"Name: {chunk.name}",
            f"File: {Path(chunk.file_path).name}"
        ])
        
        # Add docstring if available
        if chunk.docstring and chunk.docstring.strip():
            parts.append(f"Documentation: {chunk.docstring.strip()}")
        
        # Add code content
        parts.append(f"Code:\n{chunk.content}")
        
        return '\n'.join(parts)
    
    def add_chunks(self, new_chunks: list[CodeChunk]) -> int:
        """Add new code chunks and compute their embeddings.
        
        Returns:
            Number of chunks actually added (after deduplication)
        """
        if not new_chunks:
            return 0
        
        # Filter out duplicates
        unique_chunks = []
        for chunk in new_chunks:
            chunk_hash = self._chunk_hash(chunk)
            if chunk_hash not in self.chunk_hashes:
                unique_chunks.append(chunk)
                self.chunk_hashes.add(chunk_hash)
        
        if not unique_chunks:
            logger.info("No new chunks to add (all were duplicates)")
            return 0
        
        logger.info(f"Adding {len(unique_chunks)} new code chunks...")
        
        try:
            # Create text representations for embedding
            chunk_texts = [self._create_chunk_text(chunk) for chunk in unique_chunks]
            
            # Compute embeddings
            logger.info("Computing embeddings...")
            new_embeddings = self.model.encode(
                chunk_texts, 
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=False  # We'll handle normalization in FAISS
            )
            
            # Add to existing data
            self.chunks.extend(unique_chunks)
            
            if self.embeddings is None:
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            # Rebuild FAISS index
            self._build_faiss_index()
            
            # Save to disk
            self._save_data()
            
            logger.info(f"Added {len(unique_chunks)} chunks. Total: {len(self.chunks)}")
            return len(unique_chunks)
            
        except Exception as e:
            logger.error(f"Error adding chunks: {e}", exc_info=True)
            # Rollback changes
            self.chunks = self.chunks[:-len(unique_chunks)]
            for chunk in unique_chunks:
                self.chunk_hashes.discard(self._chunk_hash(chunk))
            if self.embeddings is not None and len(unique_chunks) > 0:
                self.embeddings = self.embeddings[:-len(unique_chunks)]
            raise
    
    def _build_faiss_index(self):
        """Build FAISS index for fast similarity search using cosine similarity."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return
        
        try:
            dimension = self.embeddings.shape[1]
            
            # Use L2 index with normalized embeddings for cosine similarity
            self.index = faiss.IndexFlatL2(dimension)
            
            # Normalize embeddings for cosine similarity
            normalized_embeddings = self.embeddings.copy().astype(np.float32)
            faiss.normalize_L2(normalized_embeddings)  # In-place normalization
            
            # Ensure 2D shape
            if len(normalized_embeddings.shape) == 1:
                normalized_embeddings = normalized_embeddings.reshape(1, -1)
            
            # Add embeddings to index - SWIG interface needs n and x parameters
            n_vectors = normalized_embeddings.shape[0]
            self.index.add(n_vectors, normalized_embeddings)
            
            logger.info(f"Built FAISS index with {self.index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Error building FAISS index: {e}", exc_info=True)
            self.index = None
            raise
    
    def _save_data(self):
        """Save chunks and embeddings to disk."""
        try:
            # Save chunks as JSON
            chunks_data = [self._chunk_to_dict(chunk) for chunk in self.chunks]
            with open(self.chunks_file, 'w', encoding='utf-8') as f:
                json.dump(chunks_data, f, indent=2, ensure_ascii=False)
            
            # Save embeddings
            if self.embeddings is not None:
                np.save(self.embeddings_file, self.embeddings)
            
            # Save FAISS index
            if self.index is not None:
                faiss.write_index(self.index, str(self.index_file))
            
            logger.info("Data saved to disk")
            
        except Exception as e:
            logger.error(f"Error saving data: {e}", exc_info=True)
            raise

    def search(self, query: str, k: int = 10, score_threshold: float = 0.0) -> list[tuple[CodeChunk, float]]:
        """Search for similar code chunks using semantic similarity.
        
        Args:
            query: Search query
            k: Number of results to return
            score_threshold: Minimum similarity score (cosine similarity)
            
        Returns:
            List of (chunk, similarity_score) tuples, sorted by similarity
        """
        if not self.chunks or self.index is None:
            return []
        
        try:
            # Embed the query
            query_embedding = self.model.encode([query], convert_to_numpy=True)
            faiss.normalize_L2(query_embedding)  # Normalize for cosine similarity
            
            # Ensure query is right shape and type
            query_vec = query_embedding.astype(np.float32)
            if len(query_vec.shape) == 1:
                query_vec = query_vec.reshape(1, -1)
            
            k_search = min(k, len(self.chunks))
            
            # Search using FAISS SWIG interface - need to allocate output arrays
            distances = np.empty((1, k_search), dtype=np.float32)
            labels = np.empty((1, k_search), dtype=np.int64)
            
            # SWIG interface: search(n, x, k, distances, labels)
            self.index.search(1, query_vec, k_search, distances, labels)
            
            # Convert L2 distances to cosine similarities
            # For normalized vectors: cosine_sim = 1 - (L2_distance^2 / 2)
            similarities = 1 - (distances[0] ** 2) / 2
            
            results = []
            for similarity, idx in zip(similarities, labels[0]):
                if 0 <= idx < len(self.chunks) and similarity >= score_threshold:
                    results.append((self.chunks[idx], float(similarity)))
            
            return results
            
        except Exception as e:
            logger.error(f"Error during search: {e}", exc_info=True)
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the embedding system."""
        if not self.chunks:
            return {'total_chunks': 0}
        
        stats = {
            'total_chunks': len(self.chunks),
            'chunk_types': {},
            'files': set(),
            'languages': set(),
            'embedding_dimension': self.embeddings.shape[1] if self.embeddings is not None else 0,
            'index_ready': self.index is not None
        }
        
        for chunk in self.chunks:
            # Count chunk types
            chunk_type = chunk.chunk_type
            stats['chunk_types'][chunk_type] = stats['chunk_types'].get(chunk_type, 0) + 1
            
            # Track files
            stats['files'].add(chunk.file_path)
            
            # Track languages
            file_ext = Path(chunk.file_path).suffix.lower()
            if file_ext:
                stats['languages'].add(file_ext)
        
        # Convert sets to sorted lists
        stats['total_files'] = len(stats['files'])
        stats['languages'] = sorted(list(stats['languages']))
        stats['files'] = sorted(list(stats['files']))
        
        return stats
    
    def clear_data(self):
        """Clear all stored data."""
        try:
            self._reset_data()
            
            # Remove files
            for file_path in [self.chunks_file, self.embeddings_file, self.index_file]:
                if file_path.exists():
                    file_path.unlink()

            logger.info("All data cleared")

        except Exception as e:
            logger.error(f"Error clearing data: {e}", exc_info=True)

    def remove_chunks_by_file(self, file_path: str) -> int:
        """Remove all chunks from a specific file.
        
        Returns:
            Number of chunks removed
        """
        if not self.chunks:
            return 0
        
        original_count = len(self.chunks)
        
        # Filter out chunks from the specified file
        remaining_chunks = []
        remaining_indices = []
        
        for i, chunk in enumerate(self.chunks):
            if chunk.file_path != file_path:
                remaining_chunks.append(chunk)
                remaining_indices.append(i)
        
        removed_count = original_count - len(remaining_chunks)
        
        if removed_count > 0:
            self.chunks = remaining_chunks
            self._update_chunk_hashes()
            
            # Update embeddings
            if self.embeddings is not None and remaining_indices:
                self.embeddings = self.embeddings[remaining_indices]
            elif not remaining_indices:
                self.embeddings = None
            
            # Rebuild index and save
            self._build_faiss_index()
            self._save_data()
            
            logger.info(f"Removed {removed_count} chunks from {file_path}")
        
        return removed_count


# if __name__ == "__main__":
#     # Test the embedding system
#     from src.code.parser import CodeParser
    
#     try:
#         embedding_system = EmbeddingSystem()
#         parser = CodeParser()
        
#         # Parse this file as a test
#         current_file = Path(__file__)
#         chunks = parser.parse_file(current_file)
#         added_count = embedding_system.add_chunks(chunks)
        
#         print(f"Added {added_count} chunks")
#         print("Stats:", embedding_system.get_stats())
        
#         # Test search
#         results = embedding_system.search("how to save data", k=3)
#         print(f"\nSearch results for 'how to save data':")
#         for chunk, score in results:
#             print(f"  Score: {score:.3f} - {chunk.name} ({chunk.chunk_type})")
            
#     except Exception as e:
#         print(f"Error in main: {e}")
#         raise

import ast
import re
from pathlib import Path
from typing import List, Optional
from .dataclasses import CodeChunk
from .enums import ChunkType
from ..config.logging import get_logger

logger = get_logger(__name__)

class CodeParser:
    def __init__(self):
        self.parsers = {
            '.py': self._parse_python,
            '.js': self._parse_javascript,
            '.ts': self._parse_typescript,
            '.jsx': self._parse_javascript,
            '.tsx': self._parse_typescript
        }
    
    def parse_file(self, file_path: Path) -> List[CodeChunk]:
        """Parse a single file and extract code chunks."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            parser_func = self.parsers.get(file_path.suffix)
            if parser_func:
                return parser_func(content, str(file_path))
            else:
                return []
        except Exception as e:
            logger.error(f"Error parsing {file_path}: {e}", exc_info=True)
            return []
    
    def _parse_python(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse Python file using AST."""
        chunks = []
        try:
            tree = ast.parse(content)
            lines = content.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    chunk = self._extract_python_function(node, lines, file_path)
                    if chunk:
                        chunks.append(chunk)
                elif isinstance(node, ast.ClassDef):
                    chunk = self._extract_python_class(node, lines, file_path)
                    if chunk:
                        chunks.append(chunk)
                    
                    # Extract methods from the class
                    for method_node in node.body:
                        if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            method_chunk = self._extract_python_method(
                                method_node, lines, file_path, node.name
                            )
                            if method_chunk:
                                chunks.append(method_chunk)
        except SyntaxError as e:
            logger.error(f"Syntax error in {file_path}: {e}", exc_info=True)
        
        return chunks
    
    def _extract_python_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str], 
                                file_path: str) -> Optional[CodeChunk]:
        """Extract Python function as code chunk."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Get function content
        function_lines = lines[start_line-1:end_line]
        content = '\n'.join(function_lines)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        return CodeChunk(
            content=content,
            chunk_type=ChunkType.FUNCTION,
            name=node.name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring
        )
    
    def _extract_python_class(self, node: ast.ClassDef, lines: List[str], 
                             file_path: str) -> Optional[CodeChunk]:
        """Extract Python class as code chunk."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Get class content
        class_lines = lines[start_line-1:end_line]
        content = '\n'.join(class_lines)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        return CodeChunk(
            content=content,
            chunk_type=ChunkType.CLASS,
            name=node.name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring
        )
    
    def _extract_python_method(self, node: ast.FunctionDef | ast.AsyncFunctionDef, lines: List[str], 
                              file_path: str, class_name: str) -> Optional[CodeChunk]:
        """Extract Python method as code chunk."""
        start_line = node.lineno
        end_line = node.end_lineno or start_line
        
        # Get method content
        method_lines = lines[start_line-1:end_line]
        content = '\n'.join(method_lines)
        
        # Extract docstring
        docstring = ast.get_docstring(node)
        
        return CodeChunk(
            content=content,
            chunk_type=ChunkType.METHOD,
            name=node.name,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            docstring=docstring,
            parent_class=class_name
        )
    
    def _parse_javascript(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse JavaScript/JSX using regex patterns."""
        return self._parse_js_like(content, file_path)
    
    def _parse_typescript(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse TypeScript/TSX using regex patterns."""
        return self._parse_js_like(content, file_path)
    
    def _parse_js_like(self, content: str, file_path: str) -> List[CodeChunk]:
        """Parse JavaScript-like languages using regex."""
        chunks = []
        lines = content.split('\n')
        
        # Function patterns
        patterns = [
            # Regular functions: function name() {}
            r'^(\s*)(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)\s*\{',
            # Arrow functions: const name = () => {}
            r'^(\s*)(?:export\s+)?const\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>\s*\{',
            # Method definitions: name() {}
            r'^(\s*)(?:async\s+)?(\w+)\s*\([^)]*\)\s*\{',
            # Class definitions: class Name {}
            r'^(\s*)(?:export\s+)?class\s+(\w+)(?:\s+extends\s+\w+)?\s*\{',
        ]
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            for pattern in patterns:
                match = re.match(pattern, line)
                if match:
                    indent, name = match.groups()
                    start_line = i + 1
                    
                    # Find matching closing brace
                    brace_count = 1
                    j = i + 1
                    while j < len(lines) and brace_count > 0:
                        for char in lines[j]:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    break
                        j += 1
                    
                    end_line = j
                    function_content = '\n'.join(lines[i:j])
                    
                    # Determine chunk type
                    chunk_type = ChunkType.CLASS if 'class' in line else ChunkType.FUNCTION
                    
                    chunks.append(CodeChunk(
                        content=function_content,
                        chunk_type=chunk_type,
                        name=name,
                        file_path=file_path,
                        start_line=start_line,
                        end_line=end_line
                    ))
                    
                    i = j - 1
                    break
            i += 1
        
        return chunks

if __name__ == "__main__":
    parser = CodeParser()
    test_file = Path("code_parser.py")
    chunks = parser.parse_file(test_file)
    
    logger.info(f"Found {len(chunks)} code chunks in {test_file}:")
    for chunk in chunks:
        logger.info(f"  {chunk.chunk_type}: {chunk.name} ({chunk.start_line}-{chunk.end_line})")
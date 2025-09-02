import os
import fnmatch
from pathlib import Path
from typing import List, Set
from .ignore_patterns import IGNORE_PATTERNS

class CodeScanner:
    def __init__(self):
        self.supported_extensions = {'.py', '.js', '.ts', '.jsx', '.tsx'}
        self.ignore_patterns = IGNORE_PATTERNS
    
    def should_ignore_path(self, path: Path) -> bool:
        """Check if path should be ignored based on ignore patterns."""
        path_str = str(path)
        path_parts = path.parts
        
        for pattern in self.ignore_patterns:
            if pattern in path_parts:
                return True
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(path.name, pattern):
                return True
        
        return False
    
    def scan_directory(self, directory: str) -> List[Path]:
        """Scan directory for supported code files, respecting ignore patterns."""
        directory_path = Path(directory)
        if not directory_path.exists():
            raise ValueError(f"Directory {directory} does not exist")
        
        code_files = []
        
        for root, dirs, files in os.walk(directory_path):
            root_path = Path(root)
            
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self.should_ignore_path(root_path / d)]
            
            for file in files:
                file_path = root_path / file
                
                # Check if file has supported extension
                if file_path.suffix in self.supported_extensions:
                    # Check if file should be ignored
                    if not self.should_ignore_path(file_path):
                        code_files.append(file_path)
        
        return sorted(code_files)
    
    def scan_multiple_directories(self, directories: List[str]) -> List[Path]:
        """Scan multiple directories for code files."""
        all_files = []
        seen_files = set()
        
        for directory in directories:
            try:
                files = self.scan_directory(directory)
                for file_path in files:
                    # Avoid duplicates
                    absolute_path = file_path.resolve()
                    if absolute_path not in seen_files:
                        seen_files.add(absolute_path)
                        all_files.append(file_path)
            except ValueError as e:
                print(f"Warning: {e}")
        
        return sorted(all_files)

if __name__ == "__main__":
    scanner = CodeScanner()
    files = scanner.scan_directory(".")
    print(f"Found {len(files)} code files:")
    for file in files[:10]:  # Show first 10
        print(f"  {file}")
    if len(files) > 10:
        print(f"  ... and {len(files) - 10} more")
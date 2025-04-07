#!/usr/bin/env python3
"""
Count total lines of code in the project.
This script traverses directories recursively and counts lines of code in files
with specified extensions.
"""

import os
import argparse
from typing import List, Dict, Set


def count_lines_in_file(file_path: str) -> int:
    """Count the number of lines in a file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return sum(1 for _ in f)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0


def should_include_file(file_path: str, extensions: Set[str], excluded_dirs: Set[str]) -> bool:
    """Check if file should be included in count based on extension and path."""
    # Check if file is in excluded directory
    for excluded_dir in excluded_dirs:
        if excluded_dir in file_path.split(os.sep):
            return False
    
    # Check file extension
    _, ext = os.path.splitext(file_path)
    return ext.lower() in extensions


def count_lines_recursive(directory: str, extensions: Set[str], excluded_dirs: Set[str]) -> Dict[str, Dict[str, int]]:
    """
    Recursively count lines of code in the specified directory.
    
    Args:
        directory: Directory to search in
        extensions: Set of file extensions to include (with dot, e.g. {'.py', '.json'})
        excluded_dirs: Set of directory names to exclude
        
    Returns:
        Dictionary with statistics by extension
    """
    results = {}
    file_counts = {}
    
    for ext in extensions:
        results[ext] = 0
        file_counts[ext] = 0
    
    for root, dirs, files in os.walk(directory):
        # Skip excluded directories
        dirs[:] = [d for d in dirs if d not in excluded_dirs]
        
        for file in files:
            file_path = os.path.join(root, file)
            _, ext = os.path.splitext(file_path)
            
            if ext.lower() in extensions:
                line_count = count_lines_in_file(file_path)
                results[ext.lower()] += line_count
                file_counts[ext.lower()] += 1
    
    return {'lines': results, 'files': file_counts}


def main():
    parser = argparse.ArgumentParser(description='Count lines of code in the project')
    parser.add_argument('--directory', '-d', default='.', help='Root directory to search')
    parser.add_argument('--extensions', '-e', default='.py,.json,.md,.yml,.yaml,.html,.css,.js',
                       help='Comma-separated list of file extensions to count')
    parser.add_argument('--exclude', '-x', default='venv,__pycache__,.git,.idea,node_modules',
                       help='Comma-separated list of directories to exclude')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed statistics')
    
    args = parser.parse_args()
    
    extensions = {ext.lower() if ext.startswith('.') else f'.{ext.lower()}' 
                 for ext in args.extensions.split(',')}
    excluded_dirs = set(args.exclude.split(','))
    
    print(f"Counting lines of code in '{args.directory}'...")
    print(f"Including extensions: {', '.join(sorted(extensions))}")
    print(f"Excluding directories: {', '.join(sorted(excluded_dirs))}")
    
    results = count_lines_recursive(args.directory, extensions, excluded_dirs)
    
    # Print detailed results
    if args.verbose:
        print("\nDetailed statistics:")
        print("=" * 40)
        for ext in sorted(extensions):
            if results['lines'][ext] > 0:
                print(f"{ext:8} {results['files'][ext]:6} files {results['lines'][ext]:8} lines")
    
    # Print summary
    total_files = sum(results['files'].values())
    total_lines = sum(results['lines'].values())
    
    print("\nSummary:")
    print("=" * 40)
    print(f"Total files: {total_files}")
    print(f"Total lines of code: {total_lines}")


if __name__ == "__main__":
    main() 
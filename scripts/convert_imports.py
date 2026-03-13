#!/usr/bin/env python3
"""
General script to automatically convert imports from one pattern to another.

This script can handle any import conversion pattern by specifying:
- Old import pattern (e.g., nanovllm_v6)
- New import pattern (e.g., nanovllm_v7)
- Target directories to process

Usage:
    # Basic usage - convert v6 to v7 imports
    python scripts/convert_imports.py nanovllm_v6 nanovllm_v7

    # Dry run (preview changes)
    python scripts/convert_imports.py nanovllm_v6 nanovllm_v7 --dry-run

    # Specific directory only
    python scripts/convert_imports.py nanovllm_v6 nanovllm_v7 --dir src/artifacts/nanovllm_v7

    # Convert different patterns (e.g., old_module to new_module)
    python scripts/convert_imports.py old_module new_module --dir src/my_module

    # Multiple specific directories
    python scripts/convert_imports.py nanovllm_v6 nanovllm_v7 \\
        --dir src/artifacts/nanovllm_v7 \\
        --dir src/services/nanovllm_v7

    # Only match imports from specific package (e.g., only src.services)
    python scripts/convert_imports.py nanovllm_v6 nanovllm_v7 \\
        --package-pattern src.services
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple, Optional


def find_python_files(directories: List[Path]) -> List[Path]:
    """Find all Python files in the given directories recursively."""
    files = []
    for directory in directories:
        if directory.exists():
            files.extend(directory.rglob("*.py"))
    return files


def build_import_pattern(
    old_pattern: str,
    package_constraint: Optional[str] = None
) -> str:
    """
    Build regex pattern to match imports.

    Args:
        old_pattern: The pattern to search for (e.g., 'nanovllm_v6')
        package_constraint: Optional constraint on package path (e.g., 'src.services')

    Returns:
        Regex pattern string
    """
    if package_constraint:
        # Match imports within specific package structure
        # Example: src.services.nanovllm_v6 or src.artifacts.nanovllm_v6
        return rf"(from|import)\s+({re.escape(package_constraint)}\.)[\w.]*{re.escape(old_pattern)}"
    else:
        # Match any import containing the pattern
        return rf"(from|import)\s+([\w.]*{re.escape(old_pattern)}[\w.]*)"


def find_imports_to_convert(
    content: str,
    old_pattern: str,
    package_constraint: Optional[str] = None
) -> List[Tuple[int, str, str]]:
    """
    Find all imports matching the old pattern in the file content.
    Returns list of (line_number, original_line, replacement_line).
    """
    matches = []
    lines = content.split("\n")
    pattern = build_import_pattern(old_pattern, package_constraint)

    for idx, line in enumerate(lines, start=1):
        if re.search(pattern, line):
            # Replace old_pattern with new_pattern
            replacement = re.sub(re.escape(old_pattern), args.new_pattern, line)
            matches.append((idx, line, replacement))

    return matches


def convert_file(
    file_path: Path,
    old_pattern: str,
    new_pattern: str,
    package_constraint: Optional[str] = None,
    dry_run: bool = True
) -> bool:
    """
    Convert imports in a single file.
    Returns True if file was modified (or would be modified in dry-run mode).
    """
    try:
        content = file_path.read_text()
        pattern = build_import_pattern(old_pattern, package_constraint)

        # Check if file contains any matches
        if not re.search(pattern, content):
            return False

        # Find all matches with line numbers
        lines = content.split("\n")
        matches = []
        for idx, line in enumerate(lines, start=1):
            if re.search(pattern, line):
                replacement = re.sub(re.escape(old_pattern), new_pattern, line)
                matches.append((idx, line, replacement))

        if not matches:
            return False

        print(f"\n{'='*60}")
        print(f"File: {file_path}")
        print(f"Found {len(matches)} import(s) to convert:")
        print(f"{'='*60}")

        for line_num, original, replacement in matches:
            print(f"  Line {line_num}:")
            print(f"    - {original.strip()}")
            print(f"    + {replacement.strip()}")

        if not dry_run:
            # Apply the conversion
            new_content = re.sub(re.escape(old_pattern), new_pattern, content)
            file_path.write_text(new_content)
            print(f"  ✓ Converted {len(matches)} import(s)")
        else:
            print(f"  [DRY RUN] Would convert {len(matches)} import(s)")

        return True

    except Exception as e:
        print(f"  ✗ Error processing {file_path}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Convert imports from one pattern to another in Python files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "old_pattern",
        type=str,
        help="Old import pattern to search for (e.g., nanovllm_v6)",
    )
    parser.add_argument(
        "new_pattern",
        type=str,
        help="New import pattern to replace with (e.g., nanovllm_v7)",
    )
    parser.add_argument(
        "--dir",
        action="append",
        dest="directories",
        help="Directory to process (can be specified multiple times). "
             "If not provided, defaults to v7 directories.",
    )
    parser.add_argument(
        "--package-pattern",
        type=str,
        help="Optional: Only match imports within specific package (e.g., 'src.services')",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what will be changed without making actual changes",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    global args
    args = parser.parse_args()

    # Determine which directories to process
    if args.directories:
        directories = [Path(d) for d in args.directories]
    else:
        # Default to v7 directories
        base_path = Path("src")
        directories = [
            base_path / "artifacts" / "nanovllm_v7",
            base_path / "services" / "nanovllm_v7",
        ]

    print(f"Converting '{args.old_pattern}' -> '{args.new_pattern}'...")
    if args.package_pattern:
        print(f"Package constraint: {args.package_pattern}")
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified\n")

    python_files = find_python_files(directories)
    if args.verbose:
        print(f"Found {len(python_files)} Python files to scan\n")

    total_files = len(python_files)
    modified_files = 0

    for file_path in python_files:
        if convert_file(
            file_path,
            args.old_pattern,
            args.new_pattern,
            args.package_pattern,
            dry_run=args.dry_run
        ):
            modified_files += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Directories processed: {len(directories)}")
    for d in directories:
        print(f"    - {d}")
    print(f"  Total Python files scanned: {total_files}")
    print(f"  Files with matching imports: {modified_files}")
    print(f"{'='*60}")

    if args.dry_run:
        print("\nRun without --dry-run to apply these changes.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Paste registered artifact methods directly into EngineService

This script extracts method bodies from artifacts and pastes them directly into the EngineService class.
"""

import ast
import inspect
import textwrap
from pathlib import Path
from src.core.service_base import BaseService
from src.core.artifact_base import Artifact
from src.services.l1 import L1Service


import ast
import sys
from pathlib import Path
from typing import Set


class ImportCollector(ast.NodeVisitor):
    """ AST visitor to collect all import statements."""

    def __init__(self):
        self.imports: Set[str] = set()

    def visit_Import(self, node):
        """Visit import statements like 'import os'."""
        for alias in node.names:
            self.imports.add(f"import {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        """Visit from-import statements like 'from os import path'."""
        module = node.module or ""
        level = "." * node.level  # Handle relative imports

        if node.names:
            names = [alias.name for alias in node.names]
            if len(names) == 1:
                self.imports.add(f"from {level}{module} import {names[0]}")
            else:
                names_str = ', '.join(names)
                self.imports.add(f"from {level}{module} import {names_str}")
        self.generic_visit(node)


def extract_imports_from_file(path_list: list) -> Set[str]:
    """Extract all import statements from a single Python file."""
    try:
        result = set()
        for file_path in path_list: 
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Parse the AST
            tree = ast.parse(content)

            # Collect imports
            collector = ImportCollector()
            collector.visit(tree)

            imports_set = collector.imports
            result.update(imports_set)
        return result

    except Exception as e:
        print(f"Error parsing {file_path}: {e}")
        return set()


def extract_class_components(file_path, class_name):
    """
    Extract class components: imports, class definition, __init__ method, and other methods
    """
    with open(file_path, 'r') as f:
        content = f.read()

    tree = ast.parse(content)

    # Find the class
    target_class = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            target_class = node
            break

    if not target_class:
        return None, "Class not found"

    # Extract components
    init_method = None
    other_methods = []

    for item in target_class.body:
        if isinstance(item, ast.FunctionDef):
            if item.name == '__init__':
                init_method = item
            else:
                other_methods.append(item)

    # Get source lines for accurate extraction
    lines = content.split('\n')

    # Extract __init__ method body (without the def line and decorators)
    init_body_source = ""
    if init_method:
        # Get the line numbers for the method body (excluding def line)
        init_start_line = init_method.lineno
        init_end_line = init_method.end_lineno

        body_lines = []

        for line_num in range(init_start_line - 1, init_end_line):
            line = lines[line_num]
            body_lines.append(line)

        init_body_source = '\n'.join(body_lines)

    # Extract other methods
    other_methods_source = []
    for method in other_methods:
        method_source = ast.unparse(method)
        other_methods_source.append(method_source)

    # Extract class signature (everything up to the first method)
    class_start_line = target_class.lineno - 1
    first_method_line = len(lines)

    if init_method:
        first_method_line = init_method.lineno - 1
    elif other_methods:
        first_method_line = other_methods[0].lineno - 1

    class_header_lines = []
    for line_num in range(class_start_line, first_method_line):
        if line_num < len(lines):
            class_header_lines.append(lines[line_num])

    class_header = '\n'.join(class_header_lines)

    return {
        'class_header': class_header,
        'init_body': init_body_source,
        'other_methods': other_methods_source,
    }


def extract_method_body(file_path, class_name, method_name):
    """Extract the body of a method from a class in a file."""
    try:
        with open(file_path, 'r') as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Find the class
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                # Find the method
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name == method_name:
                        # Get method source
                        method_source = ast.unparse(item)
                        return method_source
                
    except Exception as e:
        print(f"Error extracting {method_name} from {file_path}: {e}")
        return None

def generate_pasted_engine_service(service):
    """Generate EngineService with artifact methods pasted directly."""
    
    base_path = Path("/home/yyx/efficient_inference/xylo-infer")
    
    all_files = [service.path]    
    artifacts = inspect.getmembers(service.artifacts, lambda a: isinstance(a, Artifact))
    
    for name, artifact in artifacts:
        all_files.append(artifact.path)
    
    artifact_table = {name: {'file': artifact.path, 
                             'class': artifact.__class__.__name__,
                             'methods': artifact.registered_methods,
                             'objects': artifact.registered_objs} for name, artifact in artifacts}
    
    imports = extract_imports_from_file(all_files)
    
    # Extract all method bodies
    method_bodies = []
    object_assignments = []
    
    for artifact_name, info in artifact_table.items():
        file_path = base_path / info['file']
        
        # Extract method bodies
        for method_name in info['methods']:
            method_body = extract_method_body(file_path, info['class'], method_name)
            if method_body:
                # Add comment indicating source
                method_bodies.append(f"# Pasted from {info['class']}.{method_name}")
                method_bodies.append(f"{method_body}")
                method_bodies.append("")  # Empty line
        
        # Extract object assignments (these will be properties, not methods)
        for obj_name in info['objects']:
            object_assignments.append(f"        self.{obj_name} = self.artifacts.{artifact_name}.{obj_name}")
    
    # Combine method bodies
    methods_code = "\n".join(method_bodies)
    methods_code = textwrap.indent(methods_code, "    ")
    objects_code = "\n".join(object_assignments)
    
    source_parsed = extract_class_components(service.path, service.__class__.__name__)
    
    source_merged = ""
    
    source_merged += source_parsed["class_header"]
    source_merged += "\n" + source_parsed["init_body"]
    source_merged += "\n" + objects_code + "\n\n"
    
    source_merged += methods_code
    
    for raw_method_code in source_parsed["other_methods"]:
        raw_method_code = textwrap.indent(raw_method_code, "    ")
        source_merged += "\n" + raw_method_code + "\n"
    
    print(source_merged)


if __name__ == "__main__":
    service = L1Service()
    generate_pasted_engine_service(service)
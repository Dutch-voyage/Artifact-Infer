"""Render a compiled Artifact-Infer registry as JSON, Mermaid, or DOT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.core.graph import filter_registry_graph, render_registry_graph
from src.core.registry import RegistryGraph


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="compiled registry JSON file")
    parser.add_argument(
        "--format",
        choices=("json", "mermaid", "dot"),
        default="mermaid",
    )
    parser.add_argument(
        "--direction",
        choices=("TB", "TD", "BT", "LR", "RL"),
        default="LR",
    )
    parser.add_argument(
        "--extension",
        action="append",
        default=[],
        help="include one extension; may be repeated",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=[],
        help="require a component tag; may be repeated",
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    raw = json.loads(args.input.read_text(encoding="utf-8"))
    graph = RegistryGraph.from_dict(raw)
    graph = filter_registry_graph(
        graph,
        extensions=args.extension,
        tags=args.tag,
    )
    rendered = render_registry_graph(
        graph,
        format=args.format,
        direction=args.direction,
    )
    if args.output is None:
        print(rendered)
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()

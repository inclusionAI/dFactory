"""Package SolDiff model and decoding sources for standalone HF inference."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

from dfactory.recipes.soldiff.generation import SolDiffGenerationConfig


class _Source(ast.NodeVisitor):
    def __init__(self, source: Path):
        self.text = source.read_text(encoding="utf-8")
        self.lines = self.text.splitlines(keepends=True)
        self.offsets = [0]
        for line in self.lines:
            self.offsets.append(self.offsets[-1] + len(line))
        self.edits: list[tuple[int, int, str]] = []

    def replace(self, node: ast.AST, text: str) -> None:
        start = self.offsets[node.lineno - 1] + node.col_offset
        end = self.offsets[node.end_lineno - 1] + node.end_col_offset
        self.edits.append((start, end, text))

    def remove(self, node: ast.AST, *, leading_comment: bool = False) -> None:
        start = node.lineno - 1
        if leading_comment and start > 0 and self.lines[start - 1].lstrip().startswith("#"):
            start -= 1
        self.edits.append((self.offsets[start], self.offsets[node.end_lineno], ""))

    def write(self, target: Path) -> None:
        self.visit(ast.parse(self.text))
        text = self.text
        for start, end, replacement in sorted(self.edits, reverse=True):
            text = text[:start] + replacement + text[end:]
        ast.parse(text)
        target.write_text(text, encoding="utf-8")


def _definitions(
    source: Path, *, include: set[str] | None = None, exclude: set[str] | frozenset[str] = frozenset()
) -> str:
    text = source.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    definitions = []
    for node in ast.parse(text).body:
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            continue
        if node.name in exclude or (include is not None and node.name not in include):
            continue
        start = min([node.lineno] + [decorator.lineno for decorator in node.decorator_list])
        definitions.append("".join(lines[start - 1 : node.end_lineno]).rstrip())
    return "\n\n\n".join(definitions)


class _InferenceModel(_Source):
    def __init__(self, source: Path, head_source: Path, interface_source: Path):
        super().__init__(source)
        self.head_definitions = _definitions(head_source, exclude={"get_p_head"})
        self.projection = _definitions(interface_source, include={"causallm_loss_function"})

    def visit_ImportFrom(self, node):
        if node.module == "dfactory.runtime.veomni.ops":
            self.replace(
                node,
                "from transformers.modeling_layers import GradientCheckpointingLayer\n"
                "from transformers.utils.logging import get_logger\n"
                "from .generation_soldiff import SolDiffGenerationConfig, SolDiffGenerationMixin",
            )
        elif node.level == 1 and node.module == "heads":
            self.remove(node)
        elif node.module == "transformers.generation":
            self.remove(node)
        elif node.module == "collections.abc":
            self.replace(
                node,
                "import json\nfrom collections.abc import Callable\nfrom dataclasses import dataclass\n"
                "from pathlib import Path",
            )
        elif node.module == "typing":
            self.replace(node, "from typing import Any, Union")

    def visit_If(self, node):
        if isinstance(node.test, ast.Call) and isinstance(node.test.func, ast.Name):
            if node.test.func.id == "is_liger_kernel_available":
                self.remove(node)
                return
        self.generic_visit(node)

    def visit_Assign(self, node):
        # Full inference models retain all positions on a single process.
        if any(isinstance(target, ast.Name) and target.id == "sp_group" for target in node.targets):
            self.remove(node, leading_comment=True)
            return
        if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name):
            if node.value.func.id == "slice_position_embedding":
                self.remove(node)
                return
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            node.value.elts.append(ast.Constant(value="SolDiffGenerationConfig"))
            self.replace(node, ast.unparse(node))
        elif any(isinstance(target, ast.Name) and target.id == "_CHECKPOINT_FOR_DOC" for target in node.targets):
            self.replace(node, self.head_definitions + "\n\n\n" + self.projection + "\n\n\n" + ast.unparse(node))

    def visit_ClassDef(self, node):
        if node.name == "Qwen3ForBDLM":
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "GenerationMixin":
                    self.replace(base, "SolDiffGenerationMixin")
        self.generic_visit(node)


class _InferenceGeneration(_Source):
    def visit_FunctionDef(self, node):
        if node.name in {"_head_config", "load_soldiff_model"}:
            self.remove(node)
            return
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module == "dfactory.models.qwen3.heads":
            self.remove(node)
        elif node.module == "pathlib":
            self.remove(node)
        elif node.module == "dataclasses":
            self.replace(
                node,
                "from collections.abc import Mapping\nfrom dataclasses import asdict, dataclass, fields, is_dataclass",
            )
        elif node.module == "torch":
            self.replace(
                node,
                "from torch import nn\n"
                "from transformers import GenerationConfig\n"
                "from transformers.generation import GenerationMixin",
            )

    def visit_Import(self, node):
        if any(alias.name == "json" for alias in node.names):
            self.remove(node)

    def visit_Assign(self, node):
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            self.replace(node, '__all__ = ["SolDiffGenerationConfig", "SolDiffGenerationMixin", "generate_soldiff"]')
        self.generic_visit(node)


def export_soldiff_code(target: Path, config: dict) -> None:
    """Copy the current inference algorithms, replacing only runtime interfaces."""
    recipe = Path(__file__).parent
    model = recipe.parents[1] / "models" / "qwen3"
    model_path = target / "modeling_soldiff.py"
    head_source = model / "heads.py"
    interface_source = recipe / "hf.py"
    _InferenceModel(model / "modeling.py", head_source, interface_source).write(model_path)
    with model_path.open("a", encoding="utf-8") as handle:
        handle.write(
            '\nQwen3ForBDLM.register_for_auto_class("AutoModelForCausalLM")\n'
            'Qwen3Model.register_for_auto_class("AutoModel")\n'
        )
    generation_path = target / "generation_soldiff.py"
    _InferenceGeneration(recipe / "generation.py").write(generation_path)
    with generation_path.open("a", encoding="utf-8") as handle:
        handle.write("\n\n" + _definitions(head_source, include={"get_p_head"}) + "\n\n\n")
        handle.write(_definitions(interface_source, include={"SolDiffGenerationMixin"}) + "\n")
    for filename in ("heads_soldiff.py", "hf_soldiff.py"):
        (target / filename).unlink(missing_ok=True)
    defaults = asdict(SolDiffGenerationConfig())
    if config.get("mask_token_id") is not None:
        defaults["mask_token_id"] = config["mask_token_id"]
    config.setdefault("soldiff_generation_config", defaults)
    (target / "p_head_config.json").write_text(
        json.dumps(config["p_head_config"], indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

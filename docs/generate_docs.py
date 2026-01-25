"""Generate llms.md and llms-full.md from the DiffusionLab package.

Walks the package tree, extracts docstrings from all public modules, classes,
and functions, and produces two Markdown files following the llms.txt spec
(https://llmstxt.org/):

- llms.md: concise summary with module listing
- llms-full.md: complete API reference with all docstrings
"""

import importlib
import inspect
import textwrap
from pathlib import Path

PACKAGE_NAME = "diffusionlab"
BASE_URL = "https://github.com/druvpai/DiffusionLab/blob/main"
PROJECT_DESCRIPTION = (
    "No-frills JAX library providing core abstractions for diffusion models."
)


def discover_modules(package_name: str, root: Path) -> list[str]:
    """Walk the filesystem to find all Python modules under the package.

    Uses the filesystem instead of ``pkgutil.walk_packages`` so that implicit
    namespace packages (directories without ``__init__.py``) are discovered.

    Args:
        package_name: The top-level package name.
        root: Repository root directory containing the package.

    Returns:
        Sorted list of fully-qualified module names.
    """
    package_dir = root / package_name
    modules: list[str] = []
    for py_file in sorted(package_dir.rglob("*.py")):
        if py_file.name == "__init__.py":
            continue
        relative = py_file.relative_to(root).with_suffix("")
        dotted = ".".join(relative.parts)
        modules.append(dotted)
    return modules


def module_to_path(module_name: str) -> str:
    """Convert a dotted module name to a file path relative to the repo root.

    Args:
        module_name: Dotted module path (e.g. ``diffusionlab.models.mlp``).

    Returns:
        Relative file path string.
    """
    return module_name.replace(".", "/") + ".py"


def get_public_members(
    mod: object,
    module_name: str,
) -> list[tuple[str, object]]:
    """Return public classes and functions defined in a module.

    Args:
        mod: The imported module object.
        module_name: The fully-qualified module name (used to filter re-exports).

    Returns:
        List of ``(name, obj)`` pairs for public members.
    """
    return [
        (name, obj)
        for name, obj in inspect.getmembers(mod)
        if not name.startswith("_")
        and (inspect.isclass(obj) or inspect.isfunction(obj))
        and obj.__module__ == module_name
    ]


def format_member_doc(name: str, obj: object) -> str:
    """Format a single class or function with its docstring.

    Args:
        name: Member name.
        obj: The class or function object.

    Returns:
        Markdown-formatted documentation string.
    """
    kind = "class" if inspect.isclass(obj) else "function"
    sig = ""
    try:
        sig = str(inspect.signature(obj))  # type: ignore
    except (ValueError, TypeError):
        pass

    header = f"### `{name}`\n\n**{kind}** `{name}{sig}`"
    doc = inspect.getdoc(obj)
    if doc:
        return f"{header}\n\n{textwrap.dedent(doc)}"
    return header


def generate_llms_txt(modules: list[str]) -> str:
    """Generate the concise llms.txt content.

    Args:
        modules: List of fully-qualified module names.

    Returns:
        llms.txt content string.
    """
    lines = [
        f"# {PACKAGE_NAME}",
        "",
        f"> {PROJECT_DESCRIPTION}",
        "",
        "## Modules",
        "",
    ]
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        members = get_public_members(mod, mod_name)
        if not members:
            continue
        path = module_to_path(mod_name)
        summary = ""
        mod_doc = inspect.getdoc(mod)
        if mod_doc:
            summary = f": {mod_doc.splitlines()[0]}"
        lines.append(f"- [{mod_name}]({BASE_URL}/{path}){summary}")

    lines.append("")
    return "\n".join(lines)


def generate_llms_full_txt(modules: list[str], readme_text: str) -> str:
    """Generate the comprehensive llms-full.txt content.

    Args:
        modules: List of fully-qualified module names.
        readme_text: Content of the project README.

    Returns:
        llms-full.txt content string.
    """
    lines = [
        f"# {PACKAGE_NAME}",
        "",
        f"> {PROJECT_DESCRIPTION}",
        "",
        "## Overview",
        "",
        readme_text.strip(),
        "",
        "## API Reference",
        "",
    ]

    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        members = get_public_members(mod, mod_name)
        if not members:
            continue

        lines.append(f"## `{mod_name}`")
        lines.append("")
        mod_doc = inspect.getdoc(mod)
        if mod_doc:
            lines.append(mod_doc)
            lines.append("")

        for name, obj in members:
            lines.append(format_member_doc(name, obj))
            lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Entry point: generate llms.md and llms-full.md in the docs/ directory."""
    root = Path(__file__).resolve().parent.parent
    docs_dir = root / "docs"
    readme_text = (root / "README.md").read_text()

    modules = discover_modules(PACKAGE_NAME, root)

    (docs_dir / "llms.md").write_text(generate_llms_txt(modules))
    print("Generated docs/llms.md")

    (docs_dir / "llms-full.md").write_text(generate_llms_full_txt(modules, readme_text))
    print("Generated docs/llms-full.md")


if __name__ == "__main__":
    main()

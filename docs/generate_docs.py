import pkgutil
import importlib.util
import sys
import inspect
import dataclasses
from pathlib import Path

# --- Helper sets for filtering ---
OBJECT_MEMBERS = set(dir(object))
DATACLASS_METHODS = {
    "__init__",
    "__repr__",
    "__eq__",
    "__lt__",
    "__le__",
    "__gt__",
    "__ge__",
    "__hash__",
    "__match_args__",
    "__dataclass_fields__",
    "__dataclass_params__",
    "__post_init__",
}
# ---


def find_modules(package_path, prefix):
    """Recursively finds modules within a package."""
    # Ensure package_path is a list for iter_modules
    if isinstance(package_path, str):
        package_path = [package_path]

    for importer, modname, ispkg in pkgutil.iter_modules(package_path):
        full_modname = f"{prefix}.{modname}"
        yield full_modname, ispkg
        if ispkg:
            # Find the path to the subpackage
            try:
                spec = importlib.util.find_spec(full_modname)
                if spec and spec.submodule_search_locations:
                    # Recursively search in the subpackage directory
                    yield from find_modules(
                        spec.submodule_search_locations, full_modname
                    )
            except ModuleNotFoundError:
                print(
                    f"Warning: Could not find spec for {full_modname}, skipping submodules.",
                    file=sys.stderr,
                )
            except (
                Exception
            ) as e:  # Catch other potential import-related errors during spec finding
                print(
                    f"Warning: Error finding spec for {full_modname}: {e}, skipping submodules.",
                    file=sys.stderr,
                )


def generate_api_md(root_package_name, output_file):
    """Generates the api.md file by scanning the package structure."""
    output_path = Path(output_file)
    try:
        spec = importlib.util.find_spec(root_package_name)
        if spec is None or spec.origin is None:
            print(
                f"Error: Could not find package '{root_package_name}'. Is it installed or in PYTHONPATH?",
                file=sys.stderr,
            )
            return
        # Get the directory containing the root package's __init__.py
        package_dir = str(Path(spec.origin).parent)

    except ModuleNotFoundError:
        print(
            f"Error: Could not find package '{root_package_name}'. Is it installed or in PYTHONPATH?",
            file=sys.stderr,
        )
        return

    output_path.parent.mkdir(
        parents=True, exist_ok=True
    )  # Ensure docs directory exists

    with open(output_path, "w") as f:
        f.write("# API Reference\n\n")
        f.write(
            "<!-- This page contains the automatically generated API reference documentation. -->\n\n\n"
        )

        # Add the root package itself first
        f.write(f"## `{root_package_name}`\n")
        f.write("```{eval-rst}\n")
        f.write(f".. automodule:: {root_package_name}\n")
        f.write("   :members:\n")
        f.write("   :show-inheritance:\n")
        f.write("```\n\n")

        # Find and add submodules
        # Gather all modules first, then sort alphabetically
        all_modules = list(find_modules([package_dir], root_package_name))
        all_modules.sort(key=lambda x: x[0])  # Sort by full module name

        for modname, ispkg in all_modules:
            # Determine heading level based on module depth
            depth = modname.count(".")
            if depth == 1:
                heading = "##"
            elif depth == 2:
                heading = "####"
            else:  # depth >= 3
                heading = "#####"

            f.write(f"{heading} `{modname}`\n")
            f.write("```{eval-rst}\n")
            f.write(f".. automodule:: {modname}\n")
            # Using consistent options; modify here if specific modules need different options
            f.write("   :members:\n")
            f.write("   :show-inheritance:\n")
            f.write("```\n\n")

    print(f"Successfully generated API documentation at '{output_path}'")


def generate_index_md(readme_path, output_file):
    """Generates the index.md file by copying README.md and adding a toctree."""
    output_path = Path(output_file)
    try:
        with open(readme_path, "r") as f_readme:
            readme_content = f_readme.read()
    except FileNotFoundError:
        print(f"Error: README.md not found at '{readme_path}'", file=sys.stderr)
        return

    toctree_blurb = """
```{toctree}
:maxdepth: 1
:caption: Contents:

api.md
```
"""
    # Ensure output directory exists (redundant if generate_api_md runs first, but safe)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f_index:
        f_index.write(readme_content)
        f_index.write("\n\n")  # Add some separation
        f_index.write(toctree_blurb)

    print(f"Successfully generated index page at '{output_path}'")


# --- New function to generate plaintext API docs ---
def generate_plaintext_api(root_package_name, output_file):
    """Generates a plaintext file containing docstrings from the package."""
    output_path = Path(output_file)
    try:
        spec = importlib.util.find_spec(root_package_name)
        if spec is None or spec.origin is None:
            print(
                f"Error: Could not find package '{root_package_name}'. Is it installed or in PYTHONPATH?",
                file=sys.stderr,
            )
            return
        package_dir = str(Path(spec.origin).parent)
    except ModuleNotFoundError:
        print(
            f"Error: Could not find package '{root_package_name}'. Is it installed or in PYTHONPATH?",
            file=sys.stderr,
        )
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_modules = list(find_modules([package_dir], root_package_name))
    all_modules.sort(key=lambda x: x[0])  # Sort by full module name

    with open(output_path, "w") as f:
        f.write(f"# Plaintext API Reference for {root_package_name}\n\n")
        f.write("<!-- This file contains automatically extracted docstrings. -->\n\n")

        # Process the root package first
        modules_to_process = [
            (root_package_name, True)
        ] + all_modules  # Add root package to the list

        for modname, _ in modules_to_process:
            try:
                module = importlib.import_module(modname)
                print(f"Processing module: {modname}")  # Debug print

                f.write(f"## Module: `{modname}`\n\n")

                module_doc = inspect.getdoc(module)
                if module_doc:
                    f.write("```\n")
                    f.write(module_doc.strip())
                    f.write("\n```\n\n")
                else:
                    f.write("*No module docstring.*\n\n")

                # Find members (classes and functions)
                members = inspect.getmembers(
                    module,
                    lambda member: inspect.isclass(member)
                    or inspect.isfunction(member),
                )
                members.sort(key=lambda x: x[0])  # Sort members alphabetically

                for name, member_obj in members:
                    # Ensure the member is defined in *this* module, not imported
                    if getattr(member_obj, "__module__", None) != modname:
                        continue

                    # Skip members starting with a single underscore (convention for private)
                    if name.startswith("_") and not name.startswith("__"):
                        continue

                    try:
                        signature = inspect.signature(member_obj)
                        sig_str = str(signature)
                    except (
                        ValueError,
                        TypeError,
                    ):  # Handle objects without signatures (like some built-ins or classes without __init__)
                        sig_str = ""

                    docstring = inspect.getdoc(member_obj)

                    if inspect.isclass(member_obj):
                        f.write(f"### Class: `{name}`\n\n")
                        if docstring:
                            f.write("```\n")
                            f.write(docstring.strip())
                            f.write("\n```\n\n")
                        else:
                            f.write("*No class docstring.*\n\n")

                        # Process methods within the class, filtering inherited ones
                        is_dc = dataclasses.is_dataclass(member_obj)
                        methods_to_skip = OBJECT_MEMBERS
                        if is_dc:
                            methods_to_skip = methods_to_skip.union(DATACLASS_METHODS)

                        class_methods = inspect.getmembers(
                            member_obj, inspect.isfunction
                        )
                        class_methods.sort(key=lambda x: x[0])

                        for meth_name, meth_obj in class_methods:
                            # Skip methods inherited from object or generated by dataclass
                            if meth_name in methods_to_skip:
                                continue

                            # Skip methods starting with a single underscore
                            if meth_name.startswith("_") and not meth_name.startswith(
                                "__"
                            ):
                                continue

                            # Ensure the method is defined in *this* module/class context, not just imported/inherited
                            # Check if the method's __qualname__ starts with the class name
                            if not getattr(meth_obj, "__qualname__", "").startswith(
                                member_obj.__name__ + "."
                            ):
                                # This helps filter methods inherited from deeper base classes unless explicitly overridden
                                # It might still include methods from mixins defined in the same module. Refine if needed.
                                continue

                            try:
                                meth_sig = str(inspect.signature(meth_obj))
                                meth_doc = inspect.getdoc(meth_obj)
                                f.write(f"#### Method: `{meth_name}{meth_sig}`\n")
                                if meth_doc:
                                    f.write("```\n" + meth_doc.strip() + "\n```\n\n")
                                else:
                                    f.write("*No method docstring.*\n\n")
                            except (ValueError, TypeError):
                                # Fallback for methods where signature/docstring retrieval fails
                                f.write(
                                    f"#### Method: `{meth_name}`\n*Could not get signature or docstring.*\n\n"
                                )

                    elif inspect.isfunction(member_obj):
                        f.write(f"### Function: `{name}{sig_str}`\n\n")
                        if docstring:
                            f.write("```\n")
                            f.write(docstring.strip())
                            f.write("\n```\n\n")
                        else:
                            f.write("*No function docstring.*\n\n")

            except ImportError as e:
                print(
                    f"Warning: Could not import module {modname}: {e}", file=sys.stderr
                )
            except Exception as e:
                print(
                    f"Warning: Error processing module {modname}: {e}", file=sys.stderr
                )

    print(f"Successfully generated plaintext API documentation at '{output_path}'")


# --- End of new function ---


if __name__ == "__main__":
    script_dir = Path(__file__).parent  # Should be docs/
    project_root = script_dir.parent  # Assumes script is in docs/

    # Add project root and potential src directory to Python path
    sys.path.insert(0, str(project_root))
    src_dir = project_root / "src"
    if src_dir.is_dir():
        sys.path.insert(0, str(src_dir))

    # Determine the root package name and location
    package_name = "diffusionlab"
    root_package = None
    if (project_root / package_name).is_dir():
        root_package = package_name
        print(f"Found package '{package_name}' at '{project_root / package_name}'")
    elif (src_dir / package_name).is_dir():
        root_package = package_name
        print(f"Found package '{package_name}' at '{src_dir / package_name}'")
    else:
        print(
            f"Error: Could not find the '{package_name}' package directory.",
            file=sys.stderr,
        )
        print(f"Searched in: {project_root} and {src_dir}", file=sys.stderr)
        sys.exit(1)

    # Define output paths
    output_api_file = script_dir / "api.md"
    output_index_file = script_dir / "index.md"
    output_plaintext_api_file = script_dir / "llms.txt"
    root_readme_file = project_root / "README.md"

    # Generate files
    print("--- Generating docs/index.md ---")
    generate_index_md(root_readme_file, output_index_file)
    print("--- Generating docs/api.md ---")
    generate_api_md(root_package, output_api_file)
    print("--- Generating docs/llms.txt ---")
    generate_plaintext_api(root_package, output_plaintext_api_file)
    print("--- Documentation generation complete ---")

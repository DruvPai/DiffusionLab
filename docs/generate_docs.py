import os
import importlib
import pkgutil
import sys

# Add the src directory to the Python path so we can import the package
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
)


def generate_docs_for_package(package_name, docs_dir="."):
    """Generate documentation files for all modules in a package.

    Args:
        package_name: The name of the package to document
        docs_dir: The directory where docs are stored (default is current directory)
    """
    # Create the API directory if it doesn't exist
    os.makedirs(os.path.join(docs_dir, "api"), exist_ok=True)

    # Import the package
    package = importlib.import_module(package_name)

    # Get all modules in the package
    modules = []
    for _, name, ispkg in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        if ispkg:
            modules.append((name, True))  # It's a subpackage
        else:
            modules.append((name, False))  # It's a module

    # Create the API index file
    with open(os.path.join(docs_dir, "api", "index.md"), "w") as f:
        f.write("# API Reference\n\n")
        f.write(
            "This section provides detailed API documentation for all modules in DiffusionLab.\n\n"
        )

        for module_name, is_pkg in modules:
            simple_name = module_name.split(".")[-1]
            title = simple_name.replace("_", " ").title()
            f.write(f"- [{title}]({simple_name}.md)\n")

    # Create individual module documentation files
    for module_name, is_pkg in modules:
        simple_name = module_name.split(".")[-1]
        title = simple_name.replace("_", " ").title()

        with open(os.path.join(docs_dir, "api", f"{simple_name}.md"), "w") as f:
            f.write(f"# {title}\n\n")
            f.write(
                f"This module contains functionality related to {title.lower()}.\n\n"
            )
            f.write(f"::: {module_name}\n")


if __name__ == "__main__":
    # When run from the docs directory, generate docs for diffusionlab
    generate_docs_for_package("diffusionlab")
    print("Documentation files generated successfully in the api/ directory.")

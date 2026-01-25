# Repository Guidelines

## Principles for Code Generation and Documentation

Code generation principles:

- Always think step-by-step to reason about the most elegant, readable, and extensible way to accomplish the task.
- If a solution is many times more efficient, it's worth strongly considering even if a little less readable.
  - An unreadable solution is never good even if it's fast.
- If a PyTorch or Jax function is potentially vectorizable, think step-by-step about how to vectorize it.
  - Vectorized implementations are strongly preferred as they're usually faster.
- For ease of readability, use type-hints (at least for class attributes and function arguments and return values).
- For Array/Tensor-valued attributes/parameters/etc, annotate the expected shape in inline comments.
- For function-valued attributes/parameters/etc, annotate the input/output types (and shapes if applicable) in inline comments.
- Be as strict as possible about type-checking where LSP/typechecking is available. Rely on type-checking instead of superfluous assertions/error catching as much as possible. In the case of Python, this likely means you have to make bespoke (data)classes/named tuples/etc, in order to make type checking painless.
- Do not use magic numbers (e.g., 1, 2, -1, 1/2, math.pi are OK, but 84 or -20.5 are not).
- Keep code files under 1000 lines, possibly by breaking up code into different classes which can go into different files.

Code documentation principles:

- Document all functions and classes. Use Google-style docstrings with type annotations.
- Classes should be documented by their purpose and all their attributes.
- All attributes/parameters/etc should be annotated by their type in the docstrings.
- All Array/Tensor-valued attributes/parameters/etc should be annotated by their type and expected shape in the docstrings.
- All function-valued attributes/parameters/etc should be annotated by their input and output types (and expected shapes, if applicable) in the docstrings.
- Make any assumptions about the input explicit and clear in the docstring and type system, especially if it is hard to test for these assumptions efficiently.
- When doing any code change that may affect outward functionality, document it in README.md.
- Consider documenting where necessary in `docs/*.md`.
- Any context for future AI agents should be included in AGENTS.md --- only edit the part below "Project-Specific Content".

Planning principles:

- If you are making a plan, you can ask _as many questions as you want_, no restriction. Just make sure to get all the details right.
- Unless otherwise specified, make the plan as detailed as possible. For example, when planning to write Python, list all classes/methods/functions and document both the basic signature and the functionality.

### Available Commands

You can use `ag` (silver searcher) and `ast-grep` for code searching/replacing.

## Language-Specific Rules:

### Python

Please use `uv` in all projects to handle Python files, etc. To add a dependency, use `uv add <dependency>` and do not edit the `pyproject.toml` directly. To run binaries installed into the virtual environment, use `uv run`, and so on. The `uv` virtual environment is contained at the `.venv` folder. The `pyproject.toml` will almost always ensure that the project is built as a package, so you should modify import statements to use absolute imports, etc..

For type annotations:

- Use T | None instead of Optional[T]
- Use S | T instead of Union[S, T]
- Use tuple[S, T] instead of Tuple[S, T] (same with list vs List, set vs Set, dict vs Dict, etc.)

Use Pathlib instead of `os` wherever possible.

You can use `ty` and `ruff` for LSP and formatting, respectively: `uvx ty` and `uvx ruff`.

Do not read `__pycache__` or `.pyc` files.

Do not ever use default arguments.

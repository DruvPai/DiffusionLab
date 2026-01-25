import hashlib
from typing import Any, Callable

import jax
from jax import Array
from jaxtyping import PyTree

from diffusionlab.typing import PRNGKey


def bcast_right(x: Array, ndim: int) -> Array:
    assert x.ndim <= ndim, "x must have at most ndim dimensions"
    return x.reshape((*x.shape, *((1,) * max(0, ndim - x.ndim))))


def bcast_left(x: Array, ndim: int) -> Array:
    assert x.ndim <= ndim, "x must have at most ndim dimensions"
    return x.reshape((*((1,) * max(0, ndim - x.ndim)), *x.shape))


def lenient_map(
    f: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
):
    """Like jax.tree.map but with a lenient structure matching.

    The PyTree structure of the output is determined by the structure of `tree`.
    The structures of `rest` are used only to determine the leaf values to be
    mapped.

    Example usage:
      a = [1.0, 2.0]
      b = (5.0, 6.0)
      c = lenient_map(lambda x, y: x+y, a, b)
      # c is [6.0, 8.0]

    If one were to use jax.tree.map directly, one would get an error because the
    structure of `a` is not the same as the structure of `b`.

    Args:
      f: The function to apply to each leaf.
      tree: The tree to map.
      *rest: Additional arguments to pass to fn.
      is_leaf: A function that takes a leaf and returns True if it should be
        mapped. If None, all leaves are mapped.

    Returns:
      The tree resulting from applying fn to each leaf in `tree`.

    Raises:
      KeyError: If the structures of `tree` and `rest` do not match.
    """
    path_vals, struct = jax.tree_util.tree_flatten_with_path(tree, is_leaf=is_leaf)
    paths, _ = zip(*path_vals)
    restructured_rest = []
    for r in rest:
        r_path_vals, r_struct = jax.tree_util.tree_flatten_with_path(r, is_leaf=is_leaf)
        r_paths, r_leaves = zip(*r_path_vals)

        if r_paths != paths:
            raise KeyError(f"Paths of the trees must match. But {paths} != {r_paths}")

        restructured_rest.append(jax.tree_util.tree_unflatten(struct, r_leaves))
        del r_struct
    return jax.tree_util.tree_map(f, tree, *restructured_rest)


def tree_map_with_key(
    f: Callable[..., Any],
    key: PRNGKey,
    tree: PyTree,
    *rest,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    """Like jax.tree.map but with a separate PRNG key for each leaf.

    Args:
      f: The function to apply to each leaf. Takes the key as the first arg, i.e.
        of the form `f(key: PRNGKey, tree_leaf: Any, *rest_leafs: Any) -> Any`.
      key: The PRNG key from which to split all the leaf-keys.
      tree: The tree to map.
      *rest: Additional arguments to pass to f.
      is_leaf: A function that takes a leaf and returns True if it should be
        mapped. If None, all leaves are mapped.

    Returns:
      The tree resulting from applying f to each leaf.

    Raises:
      KeyError: If the structures of `tree` and `rest` do not match.
    """

    def process_leaf(path, *args):
        digest = hashlib.sha256(repr(path).encode("utf-8")).digest()
        seed = int.from_bytes(digest[:4], byteorder="little", signed=False)
        leaf_key = jax.random.fold_in(key, seed)
        return f(leaf_key, *args)

    return jax.tree_util.tree_map_with_path(process_leaf, tree, *rest, is_leaf=is_leaf)

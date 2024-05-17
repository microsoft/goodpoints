'''
JAX data structure for storing points and their attributes, along with
a bunch of helper functions.

Example of a SliceablePoints:
{
    'p': nxd, positions
    's': nxd, scores
}.
It is assumed each value is an array where the first dimension is n,
the number of points. The total dimension can vary.
'''

import jax
import jax.numpy as jnp
from jax.tree_util import tree_map, register_pytree_node_class


@register_pytree_node_class
class SliceablePoints:
    def __init__(self, pytree):
        self.pytree = pytree

    def tree_flatten(self):
        leaves, treedef = jax.tree_util.tree_flatten(self.pytree)
        return leaves, treedef

    def tree_unflatten(treedef, leaves):
        pytree = jax.tree_util.tree_unflatten(treedef, leaves)
        return SliceablePoints(pytree)

    def __getitem__(self, index):
        def apply_index(x):
            return x[index]
        return tree_map(apply_index, self.pytree)

    def get(self, key):
        return self.pytree[key]

    @property
    def length(self):
        return next(iter(self.pytree.values())).shape[0]

    def subset(self, inds):
        return SliceablePoints(self[inds])

    def pad(self, m):
        new_pytree = tree_map(lambda x: jnp.concatenate(
            [x, jnp.repeat(x[0][None], m, 0)], 0
        ), self.pytree)
        return SliceablePoints(new_pytree)

    def dynamic_slice(self, start, size):
        def apply_slice(x):
            start_inds = (start,) + (0,) * (x.ndim - 1)
            size_inds = (size,) + x.shape[1:]
            return jax.lax.dynamic_slice(x, start_inds, size_inds)
        return SliceablePoints(tree_map(apply_slice, self.pytree))

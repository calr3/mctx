# Copyright 2021 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations
from typing import Any, ClassVar, Generic, TypeVar

import chex
import jax
import jax.numpy as jnp


T = TypeVar("T")


@chex.dataclass(frozen=True)
class Tree(Generic[T]):
  """State of a search tree.

  The `Tree` dataclass is used to hold and inspect search data for a batch of
  inputs. In the fields below `B` denotes the batch dimension, `N` represents
  the number of nodes in the tree, and `num_actions` is the number of discrete
  actions.

  node_visits: `[B, N]` the visit counts for each node.
  raw_values: `[B, N]` the raw value for each node.
  node_values: `[B, N]` the cumulative search value for each node.
  parents: `[B, N]` the node index for the parents for each node.
  action_from_parent: `[B, N]` action to take from the parent to reach each
    node.
  node_rewards: `[B, N]` the immediate reward of the action from the parent to
    each node.
  node_discounts: `[B, N]` the discount between `node_rewards` and
    `node_values` for each node.
  children_index: `[B, N, num_actions]` the node index of the children for each
    action. int16 when the tree has at most 32767 nodes, else int32.
  children_prior_logits: `[B, N, num_actions]` the action prior logits of each
    node.
  embeddings: `[B, N, ...]` the state embeddings of each node.
  root_invalid_actions: `[B, num_actions]` a mask with invalid actions at the
    root. In the mask, invalid actions have ones, and valid actions have zeros.
  extra_data: `[B, ...]` extra data passed to the search.

  Every node has exactly one parent, so the statistics of the edge to a child
  are stored once, on the child. `children_visits`, `children_rewards`,
  `children_discounts` and `children_values` remain available as read-only
  `[B, N, num_actions]` properties gathered through `children_index` (zero for
  unexpanded actions), but inside a search use the per-node accessors
  (`child_visits`, `qvalues`), which gather only the rows they need.
  """
  node_visits: chex.Array  # [B, N]
  raw_values: chex.Array  # [B, N]
  node_values: chex.Array  # [B, N]
  parents: chex.Array  # [B, N]
  action_from_parent: chex.Array  # [B, N]
  node_rewards: chex.Array  # [B, N]
  node_discounts: chex.Array  # [B, N]
  children_index: chex.Array  # [B, N, num_actions]
  children_prior_logits: chex.Array  # [B, N, num_actions]
  embeddings: Any  # [B, N, ...]
  root_invalid_actions: chex.Array  # [B, num_actions]
  extra_data: T  # [B, ...]

  # The following attributes are class variables (and should not be set on
  # Tree instances).
  ROOT_INDEX: ClassVar[int] = 0
  NO_PARENT: ClassVar[int] = -1
  UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):
    return self.children_index.shape[-1]

  @property
  def num_simulations(self):
    return self.node_visits.shape[-1] - 1

  @property
  def children_visits(self):
    """`[..., N, num_actions]` visit counts of each node's children."""
    return _gather_children(self.children_index, self.node_visits)

  @property
  def children_rewards(self):
    """`[..., N, num_actions]` immediate reward of each action."""
    return _gather_children(self.children_index, self.node_rewards)

  @property
  def children_discounts(self):
    """`[..., N, num_actions]` discount between reward and child value."""
    return _gather_children(self.children_index, self.node_discounts)

  @property
  def children_values(self):
    """`[..., N, num_actions]` value of the node each action leads to."""
    return _gather_children(self.children_index, self.node_values)

  def qvalues(self, indices):
    """Compute q-values for any node indices in the tree."""
    return _per_node(_unbatched_qvalues, self, indices)

  def child_visits(self, indices):
    """Visit counts of each action's child, for any node indices in the tree.

    Like `qvalues`: a scalar index on an unbatched tree gives
    `[num_actions]`, a `[B]` index on a batched tree gives `[B, num_actions]`.
    Unexpanded actions have zero visits.
    """
    return _per_node(_unbatched_child_visits, self, indices)

  def child_prior_logits(self, indices):
    """Prior logits of each action, for any node indices in the tree.

    Shaped like `child_visits`. The logits may be stored at reduced precision
    (see `search(prior_logits_dtype=...)`); they are returned upcast to the
    dtype of the node values, so the arithmetic on them is unaffected.
    """
    return _per_node(_unbatched_child_prior_logits, self, indices)

  def summary(self) -> SearchSummary:
    """Extract summary statistics for the root node."""
    # Get state and action values for the root nodes.
    chex.assert_rank(self.node_values, 2)
    value = self.node_values[:, Tree.ROOT_INDEX]  # pyrefly: ignore[bad-index]
    batch_size, = value.shape
    root_indices = jnp.full((batch_size,), Tree.ROOT_INDEX)
    qvalues = self.qvalues(root_indices)
    # Extract visit counts and induced probabilities for the root nodes.
    visit_counts = self.child_visits(root_indices).astype(value.dtype)
    total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
    visit_probs = visit_counts / jnp.maximum(total_counts, 1)
    visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)
    # Return relevant stats.
    return SearchSummary(
        visit_counts=visit_counts,
        visit_probs=visit_probs,
        value=value,
        qvalues=qvalues)


def infer_batch_size(tree: Tree) -> int:
  """Recovers batch size from `Tree` data structure."""
  if tree.node_values.ndim != 2:
    raise ValueError("Input tree is not batched.")
  chex.assert_equal_shape_prefix(jax.tree_util.tree_leaves(tree), 1)
  return tree.node_values.shape[0]


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
@chex.dataclass(frozen=True)
class SearchSummary:
  """Stats from MCTS search."""
  visit_counts: chex.Array
  visit_probs: chex.Array
  value: chex.Array
  qvalues: chex.Array


def _gather_children(children_index, node_array):
  """`node_array[children_index]`, zero where the child is unexpanded.

  Args:
    children_index: `[..., num_actions]` child node indices, possibly
      `Tree.UNVISITED`; either one node's row or all rows `[..., N, A]`.
    node_array: `[..., N]` per-node values, with the same leading batch
      dimensions as `children_index` beyond its own node rows.

  Returns:
    The children's values, shaped like `children_index`.
  """
  if node_array.ndim > 1:
    return jax.vmap(_gather_children)(children_index, node_array)
  unvisited = children_index == Tree.UNVISITED
  # Clip the UNVISITED index (-1) to 0 rather than letting it wrap; the masked
  # result does not depend on which node it reads.
  values = jnp.take(node_array, children_index, mode="clip")
  return jnp.where(unvisited, jnp.zeros((), node_array.dtype), values)


def _per_node(fn, tree, indices):
  if jnp.asarray(indices).shape:
    return jax.vmap(fn)(tree, indices)
  else:
    return fn(tree, indices)


def _unbatched_child_visits(tree: Tree, index: int) -> chex.Array:
  chex.assert_rank(tree.node_visits, 1)
  return _gather_children(
      tree.children_index[index],  # pyrefly: ignore[bad-index]
      tree.node_visits)


def _unbatched_child_prior_logits(tree: Tree, index: int) -> chex.Array:
  chex.assert_rank(tree.node_values, 1)
  logits = tree.children_prior_logits[index]  # pyrefly: ignore[bad-index]
  return logits.astype(
      jnp.promote_types(logits.dtype, tree.node_values.dtype))


def _unbatched_qvalues(tree: Tree, index: int) -> chex.Array:
  chex.assert_rank(tree.node_values, 1)
  children = tree.children_index[index]  # pyrefly: ignore[bad-index]
  return (
      _gather_children(children, tree.node_rewards)
      + _gather_children(children, tree.node_discounts)
      * _gather_children(children, tree.node_values)
  )

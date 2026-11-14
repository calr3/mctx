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
"""The optimised search must return exactly what the original one did.

This fork changes how the search loop runs and how the tree is stored (edge
statistics live on the child node; `children_index` may be int16). None of that
may change a bit of the result, so each test runs a policy here and in the
original implementation (tests/reference, a copy of upstream mctx) and compares
the action, the action weights and every field of the original tree layout.

Compare on CPU: on a GPU two runs of the same program can already differ.
"""
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import mctx
from mctx._src.tests import reference
import numpy as np

NUM_ACTIONS = 9
NUM_CHANCE = 3
NUM_STATES = 997
BATCH_SIZE = 16


def _tables(seed):
  """Fixed random successors and outputs for a toy game of NUM_STATES states.

  Discounts include 0 (terminal) and +1 (the same player moves again).
  """
  k = jax.random.split(jax.random.PRNGKey(seed), 7)
  return dict(  # pylint: disable=use-dict-literal
      logits=jax.random.normal(k[0], (NUM_STATES, NUM_ACTIONS)) * 2,
      chance_logits=jax.random.normal(k[1], (NUM_STATES, NUM_CHANCE)),
      value=jax.random.uniform(k[2], (NUM_STATES,), minval=-1, maxval=1),
      reward=jax.random.choice(
          k[3], jnp.array([-1.0, 0.0, 0.0, 0.0, 1.0]), (NUM_STATES,)),
      discount=jax.random.choice(
          k[4], jnp.array([-1.0, -1.0, 1.0, 0.0]), (NUM_STATES,)),
      root_states=jax.random.randint(k[5], (BATCH_SIZE,), 0, NUM_STATES),
      invalid=jax.random.bernoulli(
          k[6], 0.3, (BATCH_SIZE, NUM_ACTIONS)).at[:, 0].set(False))


def _successor(state, action, salt):
  return (state * 31 + action * 7 + salt) % NUM_STATES


def _run(m, policy_name, seed, num_simulations, kwargs):
  """Runs the policy of package `m` on the toy game; returns its PolicyOutput.

  `kwargs` may name a qtransform as a string, looked up in `m`.
  """
  t = _tables(seed)
  kwargs = dict(kwargs)
  if "qtransform" in kwargs:
    kwargs["qtransform"] = getattr(m, kwargs["qtransform"])
  root_states = t["root_states"]
  root = m.RootFnOutput(
      prior_logits=t["logits"][root_states],
      value=t["value"][root_states],
      embedding=root_states)

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    nxt = _successor(embedding, action, 1)
    return m.RecurrentFnOutput(
        reward=t["reward"][nxt],
        discount=t["discount"][nxt],
        prior_logits=t["logits"][nxt],
        value=t["value"][nxt]), nxt

  def decision_recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    afterstate = _successor(embedding, action, 2)
    return m.DecisionRecurrentFnOutput(
        chance_logits=t["chance_logits"][afterstate],
        afterstate_value=t["value"][afterstate]), afterstate

  def chance_recurrent_fn(params, rng_key, chance_outcome, afterstate):
    del params, rng_key
    nxt = _successor(afterstate, chance_outcome, 3)
    return m.ChanceRecurrentFnOutput(
        action_logits=t["logits"][nxt],
        value=t["value"][nxt],
        reward=t["reward"][nxt],
        discount=t["discount"][nxt]), nxt

  if policy_name == "stochastic_muzero_policy":
    fns = {"decision_recurrent_fn": decision_recurrent_fn,
           "chance_recurrent_fn": chance_recurrent_fn}
  else:
    fns = {"recurrent_fn": recurrent_fn}

  @jax.jit
  def run(rng_key):
    return getattr(m, policy_name)(
        params=(),
        rng_key=rng_key,
        root=root,
        num_simulations=num_simulations,
        invalid_actions=t["invalid"],
        **fns,
        **kwargs)

  return run(jax.random.PRNGKey(seed + 100))


# The original Tree's fields. The optimised Tree provides the same names,
# some as properties computed from per-node arrays.
TREE_FIELDS = (
    "node_visits", "raw_values", "node_values", "parents", "action_from_parent",
    "children_index", "children_prior_logits", "children_visits",
    "children_rewards", "children_discounts", "children_values", "embeddings",
    "root_invalid_actions", "extra_data")


def _assert_identical(test, ref, new):
  """Asserts two PolicyOutputs agree bit for bit."""
  np.testing.assert_array_equal(ref.action, new.action)
  test.assertEqual(ref.action_weights.dtype, new.action_weights.dtype)
  np.testing.assert_array_equal(ref.action_weights, new.action_weights)
  for field in TREE_FIELDS:
    ref_leaves = jax.tree_util.tree_flatten_with_path(
        getattr(ref.search_tree, field))[0]
    new_leaves = jax.tree_util.tree_flatten_with_path(
        getattr(new.search_tree, field))[0]
    test.assertEqual(len(ref_leaves), len(new_leaves), field)
    for (path, x), (_, y) in zip(ref_leaves, new_leaves):
      name = field + jax.tree_util.keystr(path)
      if field == "children_index":
        # Stored as int16 when it fits; the indices must still agree.
        y = y.astype(x.dtype)
      test.assertEqual(x.dtype, y.dtype, name)
      np.testing.assert_array_equal(x, y, err_msg=name)


class SearchEquivalenceTest(parameterized.TestCase):

  @parameterized.product(
      policy=[
          ("gumbel_muzero_policy", {}),
          ("gumbel_muzero_policy",
           {"max_num_considered_actions": 4, "gumbel_scale": 0.5}),
          ("gumbel_muzero_policy",
           {"qtransform": "qtransform_by_parent_and_siblings"}),
          ("muzero_policy", {}),
          ("muzero_policy", {"temperature": 0.25}),
          ("stochastic_muzero_policy", {}),
          # A max_depth below num_simulations keeps the scatter path for new
          # nodes, since a simulation can then stop on an existing node.
          ("gumbel_muzero_policy", {"max_depth": 3}),
          ("muzero_policy", {"max_depth": 2}),
          ("stochastic_muzero_policy", {"max_depth": 3}),
          ("gumbel_muzero_policy", {"max_depth": 64}),
      ],
      num_simulations=[1, 7, 32],
  )
  def test_matches_reference_search(self, policy, num_simulations):
    policy_name, kwargs = policy
    for seed in range(3):
      ref = _run(reference, policy_name, seed, num_simulations, kwargs)
      new = _run(mctx, policy_name, seed, num_simulations, kwargs)
      _assert_identical(self, ref, new)

  def test_summary_matches_reference(self):
    ref = _run(reference, "gumbel_muzero_policy", 0, 16, {})
    new = _run(mctx, "gumbel_muzero_policy", 0, 16, {})
    ref_summary = ref.search_tree.summary()
    new_summary = new.search_tree.summary()
    for field in ("visit_counts", "visit_probs", "value", "qvalues"):
      np.testing.assert_array_equal(
          getattr(ref_summary, field), getattr(new_summary, field),
          err_msg=field)

  def test_children_index_dtype(self):
    small = _run(mctx, "muzero_policy", 0, 8, {})
    self.assertEqual(small.search_tree.children_index.dtype, jnp.int16)


class LowPrecisionPriorLogitsTest(parameterized.TestCase):
  """`prior_logits_dtype=jnp.bfloat16` stores logits at reduced precision.

  The results are no longer bit-identical, so only check the storage dtype,
  that everything stays finite and the budget is spent, and that the policy
  stays close to the full-precision one.
  """

  @parameterized.parameters(
      "gumbel_muzero_policy", "muzero_policy", "stochastic_muzero_policy")
  def test_bfloat16_logits(self, policy_name):
    num_simulations = 32
    full = _run(mctx, policy_name, 0, num_simulations, {})
    low = _run(mctx, policy_name, 0, num_simulations,
               {"prior_logits_dtype": jnp.bfloat16})
    tree = low.search_tree
    self.assertEqual(tree.children_prior_logits.dtype, jnp.bfloat16)
    self.assertEqual(tree.node_values.dtype, jnp.float32)
    self.assertEqual(tree.child_prior_logits(
        jnp.zeros(BATCH_SIZE, jnp.int32)).dtype, jnp.float32)
    self.assertTrue(np.all(np.isfinite(low.action_weights)))
    self.assertTrue(np.all(np.isfinite(tree.node_values)))
    np.testing.assert_array_equal(
        tree.node_visits[:, 0], num_simulations + 1)
    np.testing.assert_allclose(
        low.action_weights, full.action_weights, atol=0.1)


if __name__ == "__main__":
  absltest.main()

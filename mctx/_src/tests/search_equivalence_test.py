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

search.py runs the root step outside the traversal loop, applies the backward
pass's writes once and writes new nodes with slice updates. None of that may
change a bit of the result, so each test runs a policy twice, once with
`search.search` swapped for the original implementation (reference_search.py),
and compares every field of the tree, the action and the action weights.

Compare on CPU: on a GPU two runs of the same program can already differ.
"""
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import mctx
from mctx._src import search
from mctx._src.tests import reference_search
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


def _run(policy_name, seed, num_simulations, kwargs):
  """Runs the policy on the toy game and returns its PolicyOutput."""
  t = _tables(seed)
  root_states = t["root_states"]
  root = mctx.RootFnOutput(
      prior_logits=t["logits"][root_states],
      value=t["value"][root_states],
      embedding=root_states)

  def recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    nxt = _successor(embedding, action, 1)
    return mctx.RecurrentFnOutput(
        reward=t["reward"][nxt],
        discount=t["discount"][nxt],
        prior_logits=t["logits"][nxt],
        value=t["value"][nxt]), nxt

  def decision_recurrent_fn(params, rng_key, action, embedding):
    del params, rng_key
    afterstate = _successor(embedding, action, 2)
    return mctx.DecisionRecurrentFnOutput(
        chance_logits=t["chance_logits"][afterstate],
        afterstate_value=t["value"][afterstate]), afterstate

  def chance_recurrent_fn(params, rng_key, chance_outcome, afterstate):
    del params, rng_key
    nxt = _successor(afterstate, chance_outcome, 3)
    return mctx.ChanceRecurrentFnOutput(
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
    return getattr(mctx, policy_name)(
        params=(),
        rng_key=rng_key,
        root=root,
        num_simulations=num_simulations,
        invalid_actions=t["invalid"],
        **fns,
        **kwargs)

  return run(jax.random.PRNGKey(seed + 100))


def _assert_identical(test, ref, new):
  ref_leaves = jax.tree_util.tree_flatten_with_path(ref)[0]
  new_leaves = jax.tree_util.tree_flatten_with_path(new)[0]
  test.assertEqual([jax.tree_util.keystr(p) for p, _ in ref_leaves],
                   [jax.tree_util.keystr(p) for p, _ in new_leaves])
  for (path, x), (_, y) in zip(ref_leaves, new_leaves):
    name = jax.tree_util.keystr(path)
    test.assertEqual(x.dtype, y.dtype, name)
    np.testing.assert_array_equal(x, y, err_msg=name)


class SearchEquivalenceTest(parameterized.TestCase):

  @parameterized.product(
      policy=[
          ("gumbel_muzero_policy", {}),
          ("gumbel_muzero_policy",
           {"max_num_considered_actions": 4, "gumbel_scale": 0.5}),
          ("gumbel_muzero_policy",
           {"qtransform": mctx.qtransform_by_parent_and_siblings}),
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
      new = _run(policy_name, seed, num_simulations, kwargs)
      with mock.patch.object(search, "search", reference_search.search):
        ref = _run(policy_name, seed, num_simulations, kwargs)
      _assert_identical(self, ref, new)


if __name__ == "__main__":
  absltest.main()

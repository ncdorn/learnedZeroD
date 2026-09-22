"""Inference-only: batched forward pass (from nn_model.predict)."""

from functools import partial

from jax import jit

from learnedzerod._internal.neural_network.nn_util import batched_forward_pass


@partial(jit, static_argnums=(2,))
def predict(input, weights, use_leaky_relu=False):
    output = batched_forward_pass(input, weights, use_leaky_relu)
    return output

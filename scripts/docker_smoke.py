"""Sampling smoke test (used in Docker): short fit of the selected spec on JAX."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "model_build"))

import build_model  # noqa: E402
from common import build_input_data  # noqa: E402
from meridian import backend  # noqa: E402
from meridian.model.model import Meridian  # noqa: E402

data = build_input_data()
mmm = Meridian(input_data=data, model_spec=build_model.make_spec("linear_strong_pool_lag4", data))
mmm.sample_posterior(n_chains=2, n_adapt=20, n_burnin=20, n_keep=20, seed=0)
post = mmm.inference_data.posterior
print(f"backend={backend.computation_backend().name} "
      f"posterior draws={post.sizes['chain']}x{post.sizes['draw']} OK")

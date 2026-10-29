"""Container check: a short fit of the production model runs on the JAX backend."""

from meridian import backend
from meridian.model.model import Meridian

from common import build_input_data
from model_spec import make_spec

data = build_input_data()
mmm = Meridian(input_data=data, model_spec=make_spec(data))
mmm.sample_posterior(n_chains=2, n_adapt=20, n_burnin=20, n_keep=20, seed=0)
post = mmm.inference_data.posterior
print(f"backend={backend.computation_backend().name} "
      f"posterior draws={post.sizes['chain']}x{post.sizes['draw']} OK")

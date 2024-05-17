'''
Bayesian logistic regression.
'''
import numpy as np
import jax
import jax.numpy as jnp
import logging
import scipy
from pathlib import Path
from functools import partial


def sigmoid_np(x):
    t = np.maximum(x, 0)
    x = x - t
    y = 0 - t
    return np.exp(x) / (np.exp(x) + np.exp(y))


class LogisticRegression:
    def __init__(self, prior, data_name, data_folder,
                 use_probit=False,
                 subsample=None):
        self.prior = prior
        self.use_probit = use_probit
        if data_name == 'covtype':
            data_path = Path(data_folder) / 'covertype.mat'
        else:
            data_path = Path(data_folder) / 'benchmarks.mat'

        data = scipy.io.loadmat(data_path)
        if data_name == 'covtype':
            data_x = data['covtype'][:, 1:]
            data_y = data['covtype'][:, 0]
            data_y[data_y == 2] = -1
        else:
            data_x = data[data_name]['x'][0][0]
            data_y = data[data_name]['t'][0][0]
            data_y = data_y.squeeze(-1)

        logging.info(f'Loaded data {data_name} with shape '
                     f'{data_x.shape}, using prior {self.prior}')

        if subsample is not None:
            data_x = data_x[:subsample]
            data_y = data_y[:subsample]
            logging.info(f'Subsampled to {subsample} data points.')

        # Pad one to data_x. d = d_x + 1.
        data_x = np.concatenate([
            data_x,
            np.ones((data_x.shape[0], 1))], axis=1) # (n, d)
        self.num_data, self.dim = data_x.shape
        self.data_aux = (data_x, data_y)

    def sample_prior(self, rng_gen):
        if self.prior == 'laplace':
            theta_start = rng_gen.laplace(size=self.dim)
        else:
            theta_start = rng_gen.normal(size=self.dim)
        return theta_start

    def grad_log_p_prior(self, theta):
        if self.prior == 'laplace':
            return -np.sign(theta)
        else:
            return -theta

    def grad_log_p_prior_jax(self, theta):
        if self.prior == 'laplace':
            return -jnp.sign(theta)
        else:
            return -theta

    def grad_log_p_model(self, theta, inds):
        data_x, data_y = self.data_aux
        X = data_x[inds] # (b, d)
        Y = data_y[inds] # (b,)

        tmp = Y * (theta * X).sum(-1)

        if self.use_probit:
            from scipy.stats import norm
            tmp = norm.pdf(tmp) / (norm.cdf(tmp)+1e-12)
        else:
            tmp = sigmoid_np(-tmp) # (b,)
        tmp = (tmp * Y)[:, None] * X
        return tmp # (b, d)

    @partial(jax.jit, static_argnums=(0,))
    def grad_log_p_model_jax(self, theta, inds):
        data_x = jnp.array(self.data_aux[0])
        data_y = jnp.array(self.data_aux[1])
        # data_x, data_y = self.data_aux
        X = data_x[inds] # (b, d)
        Y = data_y[inds] # (b,)

        tmp = Y * (theta * X).sum(-1)
        if self.use_probit:
            from jax.scipy.stats import norm
            tmp = norm.pdf(tmp) / (norm.cdf(tmp)+1e-12)
        else:
            tmp = jax.nn.sigmoid(-tmp) # (b,)
        tmp = (tmp * Y)[:, None] * X # (b, d)
        return tmp # (b, d)

    def predict(self, theta, X):
        logit = (theta * X).sum(-1)

        if self.use_probit:
            from scipy.stats import norm
            log_p = np.log(norm.cdf(logit))
        else:
            log_p = -np.logaddexp(0, -logit)

        pred = np.where(np.exp(log_p) > 0.5, 1, -1)
        return pred

    def eval_fn(self, step, theta):
        X, Y = self.data_aux
        pred = self.predict(theta, X)

        acc = (Y == pred).sum() / len(Y)
        return f'Step: {step}, Accurarcy: {acc:.4f}'

    def sample_gold(self, num_warmup, num_sample, rng_gen):
        import numpyro
        import numpyro.distributions as dist
        from numpyro.infer import MCMC, NUTS

        # Warning: HMC is slow on GPU.
        def model(X, Y):
            if self.prior == 'laplace':
                theta = numpyro.sample('theta', dist.Laplace(0, 1),
                                       sample_shape=(self.dim,))
            else:
                theta = numpyro.sample('theta', dist.Normal(0, 1),
                                       sample_shape=(self.dim,))
            logits = (theta * X).sum(-1)
            if self.use_probit:
                from jax.scipy.stats import norm
                return numpyro.sample(
                    'obs', dist.Bernoulli(
                        probs=norm.cdf(logits)),
                    obs=(Y==1))
            else:
                return numpyro.sample('obs', dist.Bernoulli(logits=logits),
                                      obs=(Y==1))
        X, Y = self.data_aux

        import contextlib
        from ..util.backend import jax_backends
        if 'cpu' in jax_backends():
            cm = jax.default_device(jax.devices('cpu')[0])
        else:
            cm = contextlib.nullcontext()

        with cm:
            mcmc = MCMC(NUTS(model=model), num_warmup=num_warmup,
                        num_samples=num_sample)
            mcmc.run(jax.random.PRNGKey(rng_gen.integers(2**31)),
                     X, Y)
            samples = mcmc.get_samples()['theta']
        samples = np.array(samples)
        return samples

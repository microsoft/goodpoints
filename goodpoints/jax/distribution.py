'''
Simple closed-form distributions and their mixtures.
'''

from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
import numpy as np
import math

class Distribution(ABC):
    @abstractmethod
    def sample(self, rng, batch_size):
        '''
        Args:
          rng: Jax's rng.

        Returns:
          (b,d), samples
        '''
        pass

    def log_p(self, x):
        '''
        Args:
          x: (d,), a single point.
        '''
        pass

    def score(self, x):
        return jax.grad(self.log_p)(x)


def gaussian_unnormalized_log_p(x, mean, cov_inv):
    x_centered = x - mean
    tmp = -0.5 * jnp.matmul(jnp.expand_dims(x_centered, -2),
                            jnp.matmul(cov_inv,
                                       jnp.expand_dims(x_centered, -1)))
    tmp = jnp.squeeze(tmp, (-2, -1))
    return tmp


def gaussian_log_Z(cov_sqrt):
    dim = cov_sqrt.shape[-1]
    log_Z = (-dim/2 * np.log(2 * np.pi) -
             np.linalg.slogdet(cov_sqrt)[1])
    return log_Z


def gaussian_sample(rng, batch_size, mean, cov_sqrt):
    dim = cov_sqrt.shape[-1]
    z = jax.random.normal(rng, (batch_size, dim))
    x = jnp.squeeze(jnp.expand_dims(cov_sqrt, 0) @
                    jnp.expand_dims(z, -1), -1)
    return x + mean


class Gaussian(Distribution):
    def __init__(self, mean, cov_sqrt):
        '''
        Args:
          mean: (D,)
          cov_sqrt: (D, D), so that covariance is (cov_sqrt @ cov_sqrt^T).
        '''
        self.dim = mean.shape[0]
        self.mean = mean
        self.cov_sqrt = cov_sqrt

        self.log_Z = gaussian_log_Z(self.cov_sqrt)
        self.cov_inv = jnp.linalg.inv(self.cov_sqrt @ self.cov_sqrt.T)

    def sample(self, rng, batch_size):
        return gaussian_sample(rng, batch_size,
                               self.mean, self.cov_sqrt)

    def log_p(self, x):
        return gaussian_unnormalized_log_p(
            x, self.mean, self.cov_inv) + self.log_Z

    def get_cov(self):
        return self.cov_sqrt @ self.cov_sqrt.T


class Mixture(Distribution):
    def __init__(self, mixtures, weights):
        '''
        Args:
          mixtures: a list of Distribution.
          weights: weights of mixture, sum up to 1.
        '''
        self.dim = mixtures[0].dim
        self.mixtures = mixtures
        self.num_mixture = len(mixtures)
        self.weights = weights
        self.logit_weights= jnp.log(weights)

        assert(self.weights.shape[0] == self.num_mixture)
        self.select_fn = jax.vmap(lambda s_all, c: s_all[:, c])

    def sample(self, rng, batch_size):
        return self.sample_with_choices(rng, batch_size)[0]

    def sample_with_choices(self, rng, batch_size):
        choices = jax.random.categorical(rng, self.logit_weights,
                                         axis=-1,
                                         shape=(batch_size,))

        rngs = jax.random.split(rng, self.num_mixture + 1)
        rng = rngs[0]
        rngs = rngs[1:]

        samples_each = []
        for i, mixture in enumerate(self.mixtures):
            samples_each.append(mixture.sample(rngs[i], batch_size))
        samples_all = jnp.stack(samples_each, -1) # (b, d, m)

        samples = self.select_fn(samples_all, choices)
        return samples, choices

    def log_p(self, x):
        # Compute log(\sum_i w_i exp(log_p_i(x))).
        log_p_each = []
        for mixture in self.mixtures:
            log_p_each.append(mixture.log_p(x))
        log_p_all = jnp.stack(log_p_each, -1) # (m,)
        log_p = jax.scipy.special.logsumexp(log_p_all,
                                            axis=-1, b=self.weights) # ()
        log_p -= math.log(self.num_mixture)
        return log_p

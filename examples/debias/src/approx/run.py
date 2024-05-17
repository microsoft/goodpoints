'''
Approximate Bayesian inference experiement runner. This is used for the
approximate inference experiment in Li et al. 2024.
'''

import hydra

from goodpoints.jax.autotune import autotune
from goodpoints.jax.kernel.scalar import imq
from goodpoints.jax.kernel.precond_stein import PrecondSteinKernel

from .problem import ApproximateBayesian
from ..util.run_main import main_template


def make_problem_fn(cfg, cache):
    return ApproximateBayesian(cfg, cache)


def make_kernel_fn(problem, cache):
    hess_fn = problem.compute_hessian
    tuned_params = autotune(problem.points, problem.scores, hess_fn=hess_fn)
    return PrecondSteinKernel(imq, M=tuned_params['M'],
                              med_sqr=tuned_params['med_sqr'])


def make_gold_params_fn(problem):
    hess_fn = problem.compute_hessian
    return autotune(problem.points_gold, None, hess_fn=hess_fn)



@hydra.main(version_base=None,
            config_path='../config',
            config_name='approx')
def main(cfg):
    main_template(cfg, make_problem_fn, make_kernel_fn, make_gold_params_fn)


if __name__ == '__main__':
    main()

'''
Tempering experiment runner. This is used for the tempering experiment
in Li et al. 2024.
'''

import hydra

from goodpoints.jax.autotune import autotune
from goodpoints.jax.kernel.scalar import imq
from goodpoints.jax.kernel.precond_stein import PrecondSteinKernel

from ..util.on_disk import OnDisk
from ..util.run_main import main_template


def make_problem_fn(cfg, cache):
    return OnDisk(cfg, cache)


def make_kernel_fn(problem, cache):
    tuned_params = autotune(problem.points, id_metric=True)
    return PrecondSteinKernel(imq, M=tuned_params['M'],
                              med_sqr=tuned_params['med_sqr'])



def make_full_kernel_fn(problem):
    full_params = autotune(problem.points_full, id_metric=True)
    return PrecondSteinKernel(imq, M=full_params['M'],
                              med_sqr=full_params['med_sqr'])


@hydra.main(version_base=None,
            config_path='../config',
            config_name='temper')
def main(cfg):
    main_template(cfg, make_problem_fn, make_kernel_fn,
                  None,
                  make_full_kernel_fn)


if __name__ == '__main__':
    main()

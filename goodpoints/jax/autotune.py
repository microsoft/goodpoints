'''
Automatic selection of the preconditioning matrix M and the
median squared distance.
'''

import numpy as np
from goodpoints.jax.dot import mat_vec_dot

def autotune(points,
             scores=None, hess_fn=None,
             cov_score=True,
             id_metric=False,
             dist_cutoff=50,
             med_sqr_n=1000):
    '''
    Args:
        points: (n, d), input points.
        scores: (n, d), scores at each point.
        hess_fn:
            callable, Hessian function that returns the hessian of log p at x.
            Its output needs to be a negative definite matrix.
        cov_score:
            bool, whether to use scores for covariance estimation if hess_fn
            is not available.
        id_metric: bool, whether to use identity metric for M.
        dist_cutoff: percentile cutoff for distance when estimating M.
        med_sqr_n: number of samples to use for median squared distance.
    '''

    # First find mode, prioritizing scores if available.
    if scores is None:
        p_mode = points.mean(0)
    else:
        # We use Euclidean norm here.
        score_norms = np.linalg.norm(scores, axis=-1)
        mode_ind = np.argmin(score_norms)
        p_mode = points[mode_ind]

    # Next we determine the metric tensor M.
    if id_metric:
        M = np.eye(points.shape[1])
        M_inv = np.eye(points.shape[1])
    else:
        if hess_fn != None:
            M_inv = -hess_fn(p_mode)
            M = np.linalg.inv(M_inv)
        else:
            dist = np.linalg.norm(points - p_mode, axis=-1) # (n,)
            cutoff = np.percentile(dist, dist_cutoff)
            mask = dist < cutoff
            if cov_score:
                M_inv = np.cov(scores[mask].T)
                M = np.linalg.inv(M_inv)
            else:
                M = np.cov(points[mask].T)
                M_inv = np.linalg.inv(M)

    # Finally we determine med_sqr.
    med_sqr = compute_med_sqr(
        points,
        M_inv=M_inv,
        n_select=med_sqr_n
    )

    return {'M': M, 'M_inv': M_inv, 'med_sqr': med_sqr}


def compute_med_sqr(points, M_inv=None, n_select=1000):
    '''
    Compute the median squared distance of the selected input points.
    '''

    n_select = min(n_select, points.shape[0])
    inds = np.linspace(0, points.shape[0] - 1, n_select, dtype=int)
    points = points[inds]

    n, d = points.shape
    diff = points[:, None] - points[None, :] # (n, n, d)
    diff = np.reshape(diff, [-1, d]) # (n*n, d)
    if M_inv is None:
        M_inv = np.eye(d)
    tmp = mat_vec_dot(M_inv, diff) # (n*n, d)
    dist_list = (tmp * diff).sum(-1) # (n*n,)
    med_sqr = np.median(dist_list)
    return med_sqr

import wandb
import numpy as np
import matplotlib.pyplot as plt
import logging

def log_thinned_samples(step,
                        samples,
                        thinned_w,
                        wandb_key,
                        sz=5,
                        figsize=6):
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # ax.set_aspect('equal')
    # ax.scatter(samples[:, 0], samples[:, 1], c='blue', s=sz)
    ax.scatter(samples[:, 0], samples[:, 1], c='green',
               alpha=thinned_w / thinned_w.max(),
               s=sz*2.5)
    wandb.log({wandb_key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_weighted_samples(step,
                         samples,
                         weights=None,
                         wandb_key='samples',
                         sz=5,
                         figsize=6,
                         style='alpha',
                         ind0=0,
                         ind1=1,
                         star_point=None):
    assert(style in ['alpha', 'cmap'])
    if weights is None:
        weights = np.ones(samples.shape[0]) / samples.shape[0]
    fig, ax = plt.subplots(figsize=(figsize, figsize))
    # ax.set_aspect('equal')
    if style == 'alpha':
        alphas = np.clip(weights / weights.max(), 0, 1)
        ax.scatter(samples[:, ind0], samples[:, ind1],
                   c='purple', alpha=alphas, s=sz)
    else:
        s = ax.scatter(samples[:, ind0], samples[:, ind1],
                   c=weights, cmap='viridis', s=sz)
        fig.colorbar(s)

    if star_point is not None:
        # Plot the star point as a star.
        ax.scatter(star_point[ind0], star_point[ind1],
                   marker='*', s=100, c='red')


    wandb.log({wandb_key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_obj_traj(step,
                 times,
                 objs, *,
                 wandb_key,
                 log_scale_x=True,
                 log_scale_y=True):
    fig, ax = plt.subplots()
    ax.plot(times, objs)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Objective')
    if log_scale_x:
        ax.set_xscale('log')
    if log_scale_y:
        ax.set_yscale('log')
    wandb.log({wandb_key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_kernel_matrix(step,
                      K,
                      wandb_key):
    eigenvalues = np.linalg.eigvalsh(K)  # Compute the eigenvalues
    eigenvalues_sorted = np.sort(eigenvalues)[::-1]

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(24, 10))

    # Heatmap of the Kernel Matrix
    axes[0].set_title("Heatmap of Kernel Matrix")
    heatmap = axes[0].imshow(K, cmap='gray', interpolation='nearest')
    fig.colorbar(heatmap, ax=axes[0])

    # Bar Plot of Sorted Eigenvalues
    axes[1].set_title("Bar Plot of Sorted Eigenvalues")
    axes[1].bar(range(1, len(eigenvalues_sorted)+1), eigenvalues_sorted)
    axes[1].set_xlabel("Eigenvalue Index")
    axes[1].set_ylabel("Eigenvalue")
    plt.tight_layout()

    wandb.log({wandb_key: wandb.Image(fig)}, step=step)
    plt.close(fig)


def log_chain(step, chain, wandb_key):
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    x = chain[:, 0]
    y = chain[:, 1]
    ax.plot(x, y, marker='o', markersize=2, linestyle='-', color='blue')
    ax.set_title("MCMC Chain Visualization")
    wandb.log({wandb_key: wandb.Image(fig)}, step=step)
    plt.close(fig)

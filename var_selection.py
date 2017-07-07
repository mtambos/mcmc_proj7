import numpy as np
from scipy.stats import (truncnorm as sp_truncnorm,
                         norm as sp_norm,
                         chi2)


def truncnorm(a, b, loc=0, scale=1):
    a, b = (a - loc)/scale, (b - loc)/scale
    return sp_truncnorm(a=a, b=b, loc=loc, scale=scale)


def gibbs(X_init, iterations, distributions, verbose=False):
    _, M = X_init.shape
    assert M == len(distributions)
    ret_val = np.zeros((iterations, M), dtype=float)
    ret_val[0] = X_init
    for t in range(1, iterations):
        X = ret_val[t - 1].copy()
        for j in range(M):
            X[j] = distributions[j](X, j)
        ret_val[t] = X
        if verbose and t % 10 == 0:
            print("{}th iteration".format(t))
            print("X: {}".format(X))
    return ret_val


def bayes_factor(β_bar, β_old, var_star, τ, υ, λ):
    """
    REVIEW!!!! Attempt at considering truncated normals.
    TODO: cover case with non-truncated normals.
    """
    σ_star = var_star ** 0.5
    bf = np.exp(β_bar**2/(2*var_star) - β_old**2/(2*τ**2))
    bf *= σ_star / τ
    bf *= sp_norm.cdf((υ - β_bar)/σ_star) - sp_norm.cdf((λ - β_bar)/σ_star)
    bf /= sp_norm.cdf((υ - β_old)/τ) - sp_norm.cdf((λ - β_old)/τ)
    return bf


def sample_β(β_bar, p, υ, λ, τ):
    if np.random.rand() <= p:
        return 0
    else:
        return truncnorm(a=λ, b=υ, loc=β_bar, scale=τ).rvs()


def sample_var(var_bar, ν, err, N):
    return (ν*var_bar + err)/chi2.rvs(ν + N)


def sample_priors(β, var, p, τ, ν, υ, λ):
    # Sample from original priors
    β = β.copy()
    for i, b in enumerate(β):
        β[i] = sample_β(b, p[i], υ[i], λ[i], τ[i])
    var = sample_var(var, ν, 0, 0)
    return β, var


def sample_step(X, y, β, var, p, τ, ν, υ, λ):
    N, M = X.shape
    probs = np.zeros(M, dtype=float)
    β = β.copy()
    for j in range(M):
        X_j = X[:, j]
        X_sq_sum = (X_j**2).sum()
        β_no_j = β.copy()
        β_no_j[j] = 0
        β_no_j = np.atleast_2d(β_no_j)

        z = y - (X.dot(β_no_j.T)).flatten()
        b = (X_j * z).sum() / X_sq_sum
        ω_sq = var / X_sq_sum

        τ_sq = τ[j]**2
        var_star = 1/(1/ω_sq + 1/τ_sq)
        β_bar = var_star*(b/ω_sq + β[j]/τ_sq)

        bf = bayes_factor(β_bar, β[j], var_star, τ[j], υ[j], λ[j])
        p_bar = p[j]/(p[j] + (1-p[j])*bf)
        # p[j] = p_bar
        probs[j] = p_bar
        β[j] = sample_β(β_bar, p_bar, υ[j], λ[j], var_star)

    err = y - (X.dot(np.atleast_2d(β).T)).flatten()
    err = err.T.dot(err)
    var = sample_var(var, ν, err, N)
    return β, var, probs, err


def variable_selection(X, y, β, var, p, τ, ν, υ, λ, iterations, verbose=False):
    """
    REVIEW!!!! Attempt at considering truncated normals.
    TODO: cover case with non-truncated normals.
    """
    N, M = X.shape
    y = y.flatten()
    β = β.flatten().astype(float)
    chain = np.zeros((iterations, M + 1), dtype=float)
    probs = np.zeros((iterations, M), dtype=float)
    p = p.copy().astype(float)

    β, var = sample_priors(β, var, p, τ, ν, υ, λ)

    # Run sampler
    for i in range(iterations):
        β, var, probs, err = sample_step(X, y, β, var, p, τ, ν, υ, λ)
        chain[i, :M] = β
        chain[i, M] = var

        if verbose and i % (iterations/verbose) == 0:
            print("{}th iteration".format(i))
            print("Error: {}".format(err))
            print("β: {}".format(β))
            print("σ²: {}, σ: {}".format(var, var**0.5))
            print()

    return β, var, chain, probs

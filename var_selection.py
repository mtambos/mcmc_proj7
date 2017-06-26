from functools import partial
from typing import Callable, List
import warnings

import numpy as np
import pandas as pd
from scipy.stats import (multivariate_normal as mnorm,
                         truncnorm as sp_truncnorm,
                         norm as sp_norm,
                         chi2)


def truncnorm(a, b, loc=0, scale=1):
    a, b = (a - loc)/scale, (b - loc)/scale
    return sp_truncnorm(a=a, b=b, loc=loc, scale=scale)


def gibbs(X_init: np.ndarray, iterations: int, distributions: List[Callable]) -> np.ndarray:
    _, M = X_init.shape
    assert M == len(distributions)
    ret_val = np.zeros((iterations, M))
    ret_val[0] = X_init
    for t in range(1, iterations):
        X = ret_val[t - 1].copy()
        for j in range(M):
            X[j] = distributions[j](X, j)
        ret_val[t] = X
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


def variable_selection(X, y, β, var, p, τ, ν, υ, λ, iterations):
    """
    REVIEW!!!! Attempt at considering truncated normals.
    TODO: cover case with non-truncated normals.
    """
    N, M = X.shape
    y = y.flatten()
    β = β.flatten()
    chain = np.zeros((iterations, M + 1))
    probs = np.zeros((iterations, M))
    p = p.copy()
    for i in range(iterations):
        for j in range(M):
            X_j = X[:, j]
            X_sq_sum = (X_j**2).sum()
            β_no_j = β.copy()
            β_no_j[j] = 0
            z = y - (X @ np.atleast_2d(β_no_j).T).flatten()
            b = (X_j * z).sum() / X_sq_sum
            ω_sq = var / X_sq_sum
            var_star = 1/(1/ω_sq + 1/τ[j]**2)
            β_bar = var_star*(b/ω_sq + β[j]/(τ[j]**2))
            bf = bayes_factor(β_bar, β[j], var_star, τ[j], υ[j], λ[j])
            p_bar = p[j]/(p[j] + (1-p[j])*bf)
            p[j] = p_bar
            probs[i, j] = p_bar
            if np.random.rand() <= p_bar:
                β[j] = 0
            else:
                β[j] = truncnorm(a=υ[j], b=λ[j], loc=β_bar, scale=var_star).rvs()
            chain[i, j] = β[j]
        err = y - (X @ np.atleast_2d(β).T).flatten()
        err = err.T @ err
        var = (var + err)/chi2.rvs(ν + N)
        chain[i, M] = var

    return β, var, chain, probs

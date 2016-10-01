#!/usr/bin/python
# -*- coding: utf8 -*-

import numpy as np
from pandas.compat import range


def binom_interval(success, total, confint=0.95):
    """
    Compute two-sided binomial confidence interval in Python. Based on R's binom.test.
    """

    from scipy.stats import beta

    quantile = (1 - confint) / 2.
    lower = beta.ppf(quantile, success, total - success + 1)
    upper = beta.ppf(1 - quantile, success + 1, total - success)
    return (lower, upper)


def choose(n, k):
    """
    A fast way to calculate binomial coefficients by Andrew Dalke (contrib).
    """
    if np.isnan(n):
        return np.nan
    else:
        n = np.int64(n)

    if 0 <= k <= n:
        ntok = 1
        ktok = 1
        for t in range(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def class_agreement(df):
    """
    Inspired from R package e1071 matchClassed.R classAgreement
    """
    n = df.sum().sum()
    nj = df.sum(axis=0)
    ni = df.sum(axis=1)
    m = min(len(ni), len(nj))

    p0 = np.float64(np.diag(df.iloc[0:m, 0:m]).sum()) / n
    pc = np.float64((ni.iloc[0:m] * nj.iloc[0:m]).sum()) / (n**2)

    n2 = choose(n, 2)

    rand = np.float64(1) + ((df**2).sum().sum() - ((ni**2).sum() + (nj**2).sum()) / 2) / n2
    nis2 = ni[ni > 1].map(lambda x: choose(int(x), 2)).sum()
    njs2 = nj[nj > 1].map(lambda x: choose(int(x), 2)).sum()

    num = df[df > 1].dropna(axis=[0, 1], thresh=1).applymap(lambda n: choose(n, 2)).sum().sum() - np.float64(nis2 * njs2) / n2
    den = (np.float64(nis2 + njs2) / 2 - np.float64(nis2 * njs2) / n2)
    crand = num / den

    return({
        "diag": p0,
        "kappa": np.float64(p0 - pc) / (1 - pc),
        "rand": rand,
        "crand": crand
    })


def prop_test(df):
    """
    Inspired from R package caret confusionMatrix.R
    """
    from scipy.stats import binom

    x = np.diag(df).sum()
    n = df.sum().sum()
    p = (df.sum(axis=0) / df.sum().sum()).max()
    d = {
        "statistic": x,  # number of successes
        "parameter": n,  # number of trials
        "null.value": p,  # probability of success
        "p.value": binom.sf(x - 1, n, p),  # see https://en.wikipedia.org/wiki/Binomial_test
    }
    return(d)

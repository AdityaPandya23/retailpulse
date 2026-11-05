"""Distribution fits: parameters, log-likelihood, AIC and Q-Q points.

  average order value per customer   ~ lognormal (vs normal)
  repeat orders = frequency - 1      ~ Poisson vs negative binomial (dispersion ratio)
  days between purchase days         ~ exponential (vs Weibull, which equals it at shape 1)

Frequency is >= 1 by construction (every customer bought once), so the count that is fitted is
the number of repeat orders, which starts at 0 like a Poisson / negative binomial variable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import optimize, stats

from retailpulse import contracts as C

N_QQ = 200  # Q-Q points per fit (fixed plotting positions keep the artifact small)
MAX_K = 30  # observed vs expected table for k = 0..MAX_K repeat orders


def plotting_probs(n_points: int = N_QQ) -> np.ndarray:
    return (np.arange(1, n_points + 1) - 0.5) / n_points


def qq_points(
    sample: np.ndarray,
    frozen,
    variable: str,
    dist: str,
    discrete: bool = False,
    n_points: int = N_QQ,
) -> pd.DataFrame:
    """Theoretical vs sample quantiles at the same probabilities."""
    probs = plotting_probs(n_points)
    method = "inverted_cdf" if discrete else "linear"
    return pd.DataFrame(
        {
            "variable": variable,
            "dist": dist,
            "prob": probs,
            "theoretical": np.asarray(frozen.ppf(probs), dtype="float64"),
            "sample": np.quantile(sample, probs, method=method).astype("float64"),
        }
    )


def _fit(params: dict, loglik: float, n_params: int) -> dict:
    return {
        "params": {k: float(v) for k, v in params.items()},
        "loglik": float(loglik),
        "aic": float(2 * n_params - 2 * loglik),
    }


def _best(fits: dict) -> str:
    return min(fits, key=lambda name: fits[name]["aic"])


def _how_much(delta_aic: float) -> str:
    return "far better" if delta_aic > 100 else "better" if delta_aic > 10 else "slightly better"


# ---------------- order value ~ lognormal ----------------
def fit_order_value(values) -> tuple[dict, pd.DataFrame]:
    x = np.asarray(values, dtype="float64")
    x = x[np.isfinite(x) & (x > 0)]
    s, _, scale = stats.lognorm.fit(x, floc=0)
    lognormal = stats.lognorm(s, loc=0, scale=scale)
    mean, sd = stats.norm.fit(x)
    normal = stats.norm(mean, sd)
    fits = {
        "lognormal": _fit({"mu_log": np.log(scale), "sigma_log": s}, lognormal.logpdf(x).sum(), 2),
        "normal": _fit({"mean": mean, "sd": sd}, normal.logpdf(x).sum(), 2),
    }
    best = _best(fits)
    other = "normal" if best == "lognormal" else "lognormal"
    delta = fits[other]["aic"] - fits[best]["aic"]
    median, skew = float(np.median(x)), float(stats.skew(x))
    entry = {
        "title": "Average order value per customer: lognormal vs normal",
        "data": "features_full.avg_basket_value (GBP)",
        "n": int(len(x)),
        "mean": float(x.mean()),
        "median": median,
        "skewness": skew,
        "skewness_of_log": float(stats.skew(np.log(x))),
        "fits": fits,
        "best_fit": best,
        "delta_aic": float(delta),
        "conclusion": (
            f"Average order value per customer is right-skewed (median £{median:,.0f} vs mean "
            f"£{x.mean():,.0f}, skewness {skew:.1f}); a {best} fits {_how_much(delta)} than a "
            f"{other} (AIC {fits[best]['aic']:,.0f} vs {fits[other]['aic']:,.0f})"
            + (", so it is analysed on a log scale." if best == "lognormal" else ".")
        ),
    }
    qq = pd.concat(
        [
            qq_points(x, lognormal, "avg_basket_value", "lognormal"),
            qq_points(x, normal, "avg_basket_value", "normal"),
        ],
        ignore_index=True,
    )
    return entry, qq


# ---------------- repeat orders ~ Poisson vs negative binomial ----------------
def nbinom_mle(x: np.ndarray) -> tuple[float, float]:
    """MLE (size r, prob p) of a negative binomial.

    For any fixed r the MLE of the mean is the sample mean, so only r is optimised
    (profile likelihood) with p = r / (r + mean).
    """
    x = np.asarray(x)
    mu = float(x.mean())
    values, counts = np.unique(x, return_counts=True)

    def nll(log_r: float) -> float:
        r = np.exp(log_r)
        return -float((counts * stats.nbinom.logpmf(values, r, r / (r + mu))).sum())

    res = optimize.minimize_scalar(nll, bounds=(-12.0, 12.0), method="bounded")
    r = float(np.exp(res.x))
    return r, r / (r + mu)


def fit_frequency(frequency, max_k: int = MAX_K) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    x = np.asarray(frequency, dtype="int64") - 1
    n, mean, var = len(x), float(x.mean()), float(x.var(ddof=1))
    dispersion = var / mean if mean > 0 else float("nan")
    poisson = stats.poisson(mean)
    r, p = nbinom_mle(x)
    nbinom = stats.nbinom(r, p)
    fits = {
        "poisson": _fit({"mu": mean}, poisson.logpmf(x).sum(), 1),
        "nbinom": _fit(
            {"size_r": r, "prob_p": p, "mu": mean, "alpha": 1 / r}, nbinom.logpmf(x).sum(), 2
        ),
    }
    best = _best(fits)
    delta = abs(fits["poisson"]["aic"] - fits["nbinom"]["aic"])

    k = np.arange(max_k + 1)
    observed = np.bincount(np.clip(x, 0, max_k + 1), minlength=max_k + 2)[: max_k + 1]
    table = pd.DataFrame(
        {"k": k, "observed": observed, "poisson": n * poisson.pmf(k), "nbinom": n * nbinom.pmf(k)}
    )
    if best == "nbinom":
        conclusion = (
            f"Repeat orders per customer (mean {mean:.1f}) are overdispersed: variance/mean = "
            f"{dispersion:.1f} where a Poisson needs 1, and the negative binomial fits "
            f"{_how_much(delta)} (AIC {fits['nbinom']['aic']:,.0f} vs "
            f"{fits['poisson']['aic']:,.0f}), so customers differ strongly in how often they buy."
        )
    else:
        conclusion = (
            f"Repeat orders per customer (mean {mean:.1f}, variance/mean = {dispersion:.2f}) are "
            f"consistent with a Poisson: the negative binomial does not improve the fit "
            f"(AIC {fits['poisson']['aic']:,.0f} vs {fits['nbinom']['aic']:,.0f})."
        )
    entry = {
        "title": "Repeat orders per customer: Poisson vs negative binomial",
        "data": "features_full.frequency - 1 (orders after the first)",
        "n": int(n),
        "mean": mean,
        "variance": var,
        "dispersion_ratio": dispersion,
        "max": int(x.max()),
        "fits": fits,
        "best_fit": best,
        "delta_aic": float(delta),
        "tail_above_max_k": {
            "k": max_k,
            "observed": int((x > max_k).sum()),
            "poisson": float(n * poisson.sf(max_k)),
            "nbinom": float(n * nbinom.sf(max_k)),
        },
        "conclusion": conclusion,
    }
    qq = pd.concat(
        [
            qq_points(x, poisson, "repeat_orders", "poisson", discrete=True),
            qq_points(x, nbinom, "repeat_orders", "nbinom", discrete=True),
        ],
        ignore_index=True,
    )
    return entry, qq, table


# ---------------- inter-purchase gap ~ exponential ----------------
def interpurchase_gaps(sales: pd.DataFrame) -> np.ndarray:
    """Days between consecutive purchase days of the same customer, pooled over customers."""
    d = sales.loc[sales["customer_id"].notna(), ["customer_id", "invoice_date"]]
    d = (
        d.assign(day=d["invoice_date"].dt.normalize())
        .drop_duplicates(["customer_id", "day"])
        .sort_values(["customer_id", "day"])
    )
    gaps = d.groupby("customer_id")["day"].diff().dt.days.dropna()
    return gaps.to_numpy(dtype="float64")


def fit_gaps(gaps) -> tuple[dict, pd.DataFrame]:
    x = np.asarray(gaps, dtype="float64")
    x = x[np.isfinite(x) & (x > 0)]
    _, scale = stats.expon.fit(x, floc=0)
    expon = stats.expon(loc=0, scale=scale)
    shape, _, w_scale = stats.weibull_min.fit(x, floc=0)
    weibull = stats.weibull_min(shape, loc=0, scale=w_scale)
    fits = {
        "exponential": _fit(
            {"scale_days": scale, "rate_per_day": 1 / scale}, expon.logpdf(x).sum(), 1
        ),
        "weibull": _fit({"shape": shape, "scale_days": w_scale}, weibull.logpdf(x).sum(), 2),
    }
    best = _best(fits)
    delta = abs(fits["exponential"]["aic"] - fits["weibull"]["aic"])
    median, mean = float(np.median(x)), float(x.mean())
    cv = float(x.std(ddof=1) / mean)
    head = (
        f"Days between a customer's purchase days (median {median:.0f}, mean {mean:.0f}, "
        f"coefficient of variation {cv:.2f}; an exponential has 1)"
    )
    if best == "weibull":
        shape_meaning = (
            "more very short gaps and a longer tail than a memoryless process"
            if shape < 1
            else "gaps are more regular than a memoryless process"
        )
        conclusion = (
            f"{head} are only roughly exponential: a Weibull with shape {shape:.2f} fits "
            f"{_how_much(delta)} (AIC {fits['weibull']['aic']:,.0f} vs "
            f"{fits['exponential']['aic']:,.0f}), i.e. {shape_meaning}."
        )
    else:
        conclusion = (
            f"{head} are well described by an exponential (AIC "
            f"{fits['exponential']['aic']:,.0f} vs Weibull {fits['weibull']['aic']:,.0f}): the "
            "chance of buying again does not depend on the time since the last purchase."
        )
    entry = {
        "title": "Inter-purchase gap: exponential vs Weibull",
        "data": "days between consecutive purchase days, pooled over customers (all sales)",
        "n_gaps": int(len(x)),
        "mean": mean,
        "median": median,
        "cv": cv,
        "fits": fits,
        "best_fit": best,
        "delta_aic": float(delta),
        "conclusion": conclusion,
    }
    qq = pd.concat(
        [
            qq_points(x, expon, "interpurchase_days", "exponential"),
            qq_points(x, weibull, "interpurchase_days", "weibull"),
        ],
        ignore_index=True,
    )
    return entry, qq


def run(
    features_full: pd.DataFrame, sales: pd.DataFrame
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """All three fits -> (metrics entries, Q-Q points, observed-vs-expected frequency table)."""
    order_entry, qq_order = fit_order_value(features_full["avg_basket_value"])
    freq_entry, qq_freq, freq_table = fit_frequency(features_full["frequency"])
    gap_entry, qq_gap = fit_gaps(interpurchase_gaps(sales))
    entries = {
        "order_value_lognormal": order_entry,
        "frequency_poisson_vs_nbinom": freq_entry,
        "gap_exponential": gap_entry,
    }
    qq = pd.concat([qq_order, qq_freq, qq_gap], ignore_index=True)
    return (
        entries,
        C.validate(qq, C.QQ_PARQUET),
        C.validate(freq_table, C.FREQUENCY_FIT_PARQUET),
    )

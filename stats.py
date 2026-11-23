"""Alpha and deflated Sharpe ratio of daily returns: alpha is what is left after least-squares exposure to reference
returns; the deflated Sharpe ratio (Bailey and Lopez de Prado 2014) uses an AR(1)-prewhitened Newey-West effective
sample. Sharpe ratios are per day inside, annualized with 365 days on the way out."""
import numpy as np
from scipy.stats import norm

EULER_GAMMA = 0.5772


def alpha(returns, references):
    """Returns minus their least-squares exposure to the references; the intercept stays in."""
    reference = np.column_stack(references)
    exposures = np.linalg.lstsq(reference - reference.mean(0), returns, rcond=None)[0]
    return returns - reference @ exposures, exposures


def deflated_sharpe(series, trials):
    """Annual Sharpe, effective sample, annual luck bar and DSR of a daily series after `trials` tries."""
    n = len(series)
    centered = series - series.mean()
    squares = centered @ centered
    effective = squares / long_run_variance(centered)
    sharpe = series.mean() / np.sqrt(squares / (n - 1))
    standardized = centered / np.sqrt(squares / n)
    skewness, kurtosis = (standardized ** 3).mean(), (standardized ** 4).mean()
    error = np.sqrt((1 - skewness * sharpe + (kurtosis - 1) / 4 * sharpe ** 2) / (effective - 1))
    luck = 0.0 if trials == 1 else ((1 - EULER_GAMMA) * norm.isf(1 / trials) +
                                    EULER_GAMMA * norm.isf(1 / (trials * np.e))) / np.sqrt(effective)
    return dict(sharpe=sharpe * np.sqrt(365), effective=effective, luck=luck * np.sqrt(365),
                dsr=norm.cdf((sharpe - luck) / error))


def long_run_variance(centered):
    """AR(1)-prewhitened Newey-West variance (Andrews and Monahan 1992, persistence capped at 0.97)."""
    persistence = np.clip(centered[1:] @ centered[:-1] / (centered[:-1] @ centered[:-1]), -0.97, 0.97)
    residuals = centered[1:] - persistence * centered[:-1]
    lags = int(np.floor(4 * (len(centered) / 100) ** (2 / 9)))
    total = residuals @ residuals
    for lag in range(1, lags + 1):
        total += 2 * (1 - lag / (lags + 1)) * (residuals[lag:] @ residuals[:-lag])
    return total / len(residuals) / (1 - persistence) ** 2

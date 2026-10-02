"""Numerical checks of the book's checkable claims, chapter by chapter.

Each test states a result the prose derives and verifies it numerically, independently of
the figure scripts (which run on import). A claim that changes in the text should change
here too.
"""

import itertools

import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, log_softmax, logsumexp

rng = np.random.default_rng(0)


# --- Ch 1: the same residual costs differently under different losses --------------------
def test_ch1_pinball_loss_values_quoted_in_caption():
    r, tau = -2.0, 0.8
    pinball = max(tau * r, (tau - 1) * r)
    assert r**2 == 4.0 and abs(r) == 2.0 and pinball == pytest.approx(0.4)


# --- Ch 3: MAP with a Gaussian prior is ridge regression -----------------------------------
def test_ch3_map_gaussian_prior_equals_ridge_closed_form():
    X = rng.normal(size=(50, 3))
    y = X @ np.array([1.0, -2.0, 0.5]) + rng.normal(scale=0.5, size=50)
    sigma2, tau2 = 0.25, 1.0
    neg_log_post = lambda w: ((y - X @ w) ** 2).sum() / (2 * sigma2) + (w**2).sum() / (2 * tau2)
    w_map = minimize(neg_log_post, np.zeros(3)).x
    lam = sigma2 / tau2
    w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(3), X.T @ y)
    np.testing.assert_allclose(w_map, w_ridge, atol=1e-4)


# --- Ch 4 / 5: Gaussian NLL -> MSE -> mean; Laplace NLL -> MAE -> median -------------------
def test_ch4_mse_minimizer_is_the_mean_and_ch5_mae_minimizer_is_the_median():
    y = rng.standard_cauchy(201)
    c_mse = minimize_scalar(lambda c: ((y - c) ** 2).mean()).x
    c_mae = minimize_scalar(lambda c: np.abs(y - c).mean(), bounds=(-50, 50), method="bounded",
                            options={"xatol": 1e-10}).x
    assert c_mse == pytest.approx(y.mean(), abs=1e-6)
    assert c_mae == pytest.approx(np.median(y), abs=1e-6)


def test_ch4_gaussian_nll_is_mse_up_to_affine_constants():
    y, f, sigma = rng.normal(size=100), rng.normal(size=100), 1.3
    nll = -(-0.5 * np.log(2 * np.pi * sigma**2) - (y - f) ** 2 / (2 * sigma**2)).sum()
    mse = ((y - f) ** 2).mean()
    assert nll == pytest.approx(len(y) / (2 * sigma**2) * mse + len(y) / 2 * np.log(2 * np.pi * sigma**2))


@pytest.mark.parametrize("tau", [0.1, 0.5, 0.8])
def test_ch1_pinball_minimizer_is_the_tau_quantile(tau):
    y = rng.normal(size=1001)
    loss = lambda c: np.maximum(tau * (y - c), (tau - 1) * (y - c)).mean()
    c = minimize_scalar(loss, bounds=(-5, 5), method="bounded", options={"xatol": 1e-10}).x
    assert c == pytest.approx(np.quantile(y, tau, method="inverted_cdf"), abs=1e-3)


def test_ch5_huber_is_quadratic_inside_delta_and_linear_outside():
    delta = 1.5
    huber = lambda r: np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))
    r = np.linspace(-6, 6, 1001)
    inside, outside = np.abs(r) <= delta, np.abs(r) > delta
    np.testing.assert_allclose(huber(r)[inside], 0.5 * r[inside] ** 2)
    np.testing.assert_allclose(np.gradient(huber(r), r)[outside & (np.abs(r) > delta + 0.1)],
                               delta * np.sign(r[outside & (np.abs(r) > delta + 0.1)]), atol=1e-6)


# --- Ch 6: cross-entropy is exactly negative log-likelihood ---------------------------------
def test_ch6_binary_and_categorical_cross_entropy_are_nll():
    z, y = rng.normal(size=200), rng.integers(0, 2, 200)
    p = expit(z)
    bce = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    bernoulli_nll = -np.log(np.where(y == 1, p, 1 - p)).mean()
    assert bce == pytest.approx(bernoulli_nll)
    logits, labels = rng.normal(size=(50, 5)), rng.integers(0, 5, 50)
    ce = -log_softmax(logits, axis=1)[np.arange(50), labels].mean()
    manual = (-logits[np.arange(50), labels] + logsumexp(logits, axis=1)).mean()
    assert ce == pytest.approx(manual)


# --- Ch 7: KL = cross-entropy - entropy >= 0 -------------------------------------------------
def test_ch7_kl_decomposition_and_gibbs_inequality():
    for _ in range(100):
        p, q = rng.dirichlet(np.ones(6)), rng.dirichlet(np.ones(6))
        H, CE, KL = -(p * np.log(p)).sum(), -(p * np.log(q)).sum(), (p * np.log(p / q)).sum()
        assert KL == pytest.approx(CE - H)
        assert KL >= 0


# --- Ch 8: the log score and the Brier score are strictly proper ----------------------------
@pytest.mark.parametrize("score", ["log", "brier"])
def test_ch8_proper_scoring_rules_are_maximized_only_at_the_truth(score):
    def expected_score(q, p):
        if score == "log":
            return (p * np.log(q)).sum()
        return sum(p[y] * -((q - np.eye(len(p))[y]) ** 2).sum() for y in range(len(p)))
    for _ in range(20):
        p = rng.dirichlet(np.ones(4))
        best = expected_score(p, p)
        for _ in range(50):
            q = rng.dirichlet(np.ones(4))
            assert expected_score(q, p) < best


# --- Ch 9: hinge and log2-logistic upper-bound the 0-1 loss -----------------------------------
def test_ch9_surrogates_upper_bound_zero_one_loss():
    m = np.linspace(-5, 5, 2001)
    zero_one = (m <= 0).astype(float)
    assert np.all(np.maximum(0, 1 - m) >= zero_one)
    assert np.all(np.log2(1 + np.exp(-m)) >= zero_one - 1e-12)


# --- Ch 11: RankNet's pull on s_i is sigma(-Delta) --------------------------------------------
def test_ch11_ranknet_gradient_is_sigmoid_of_minus_margin():
    delta = np.linspace(-6, 6, 101)
    loss = lambda d: np.log1p(np.exp(-d))
    h = 1e-6
    pull = -(loss(delta + h) - loss(delta - h)) / (2 * h)
    np.testing.assert_allclose(pull, expit(-delta), atol=1e-6)


# --- Ch 12: Plackett-Luce is a distribution over permutations; ListMLE is its NLL -------------
def test_ch12_plackett_luce_probabilities_sum_to_one():
    s = rng.normal(size=4)

    def pl_logprob(order):
        remaining, lp = list(range(4)), 0.0
        for item in order:
            lp += s[item] - logsumexp(s[remaining])
            remaining.remove(item)
        return lp
    total = sum(np.exp(pl_logprob(o)) for o in itertools.permutations(range(4)))
    assert total == pytest.approx(1.0)


# --- Ch 14: InfoNCE is (K+1)-way cross-entropy with similarities as logits ----------------------
def test_ch14_infonce_equals_categorical_cross_entropy():
    sims, temperature = rng.normal(size=9), 0.5  # index 0 is the positive, 8 negatives
    infonce = -np.log(np.exp(sims[0] / temperature) / np.exp(sims / temperature).sum())
    ce = -log_softmax(sims / temperature)[0]
    assert infonce == pytest.approx(ce)


# --- Ch 15: perplexity is exp(average per-token NLL) -----------------------------------------
def test_ch15_uniform_model_perplexity_equals_vocabulary_size():
    vocab, n = 37, 500
    nll = -np.log(np.full(n, 1 / vocab))
    assert np.exp(nll.mean()) == pytest.approx(vocab)


# --- Ch 16: the DPO loss depends only on the margin of log-ratios -----------------------------
def test_ch16_dpo_loss_is_invariant_to_shared_shifts_and_matches_bradley_terry():
    beta = 0.2
    lp_w, lp_l, ref_w, ref_l = rng.normal(size=4)

    def dpo(lp_w, lp_l):
        return -np.log(expit(beta * (lp_w - ref_w) - beta * (lp_l - ref_l)))
    assert dpo(lp_w, lp_l) == pytest.approx(dpo(lp_w + 3.7, lp_l + 3.7))
    # Same as Bradley-Terry NLL with implicit rewards r = beta * log(pi / pi_ref).
    r_w, r_l = beta * (lp_w - ref_w), beta * (lp_l - ref_l)
    assert dpo(lp_w, lp_l) == pytest.approx(-np.log(np.exp(r_w) / (np.exp(r_w) + np.exp(r_l))))

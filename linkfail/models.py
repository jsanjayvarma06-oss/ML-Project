"""The three linear classifiers of the paper, implemented from scratch in NumPy.

* LogisticRegressionNewton - logistic regression trained with Newton's method
* GDA                       - Gaussian discriminant analysis, shared covariance
* LinearSVM                 - soft-margin linear SVM, Pegasos sub-gradient solver

All follow the scikit-learn estimator interface (fit / predict /
predict_proba or decision_function) so they plug into sklearn metrics.
Labels: y = 1 -> link NOT healthy, y = 0 -> healthy.
"""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def _add_intercept(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


class _Standardiser:
    """Feature scaling kept inside the model so callers pass raw X1, X2."""

    def _fit_scale(self, X):
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0) + 1e-12

    def _scale(self, X):
        return (X - self.mu_) / self.sigma_

    def _unscale_theta(self, theta):
        """Map [b, w] learnt on scaled data back to raw-feature units."""
        w = theta[1:] / self.sigma_
        b = theta[0] - np.sum(theta[1:] * self.mu_ / self.sigma_)
        return np.concatenate([[b], w])


class LogisticRegressionNewton(ClassifierMixin, BaseEstimator, _Standardiser):
    """Minimises the average cross-entropy J(theta) (Eq. 3) with Newton steps."""

    def __init__(self, l2: float = 1e-4, max_iter: int = 50, tol: float = 1e-10):
        self.l2, self.max_iter, self.tol = l2, max_iter, tol

    def cost(self, Xb, y, theta):
        h = np.clip(sigmoid(Xb @ theta), 1e-15, 1 - 1e-15)
        return -np.mean(y * np.log(h) + (1 - y) * np.log(1 - h))

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self._fit_scale(X)
        Xb = _add_intercept(self._scale(X))
        n, d = Xb.shape
        theta = np.zeros(d)
        reg = self.l2 * np.eye(d)
        reg[0, 0] = 0.0                                   # don't penalise bias
        self.cost_history_ = [self.cost(Xb, y, theta)]
        for _ in range(self.max_iter):
            h = sigmoid(Xb @ theta)
            grad = Xb.T @ (h - y) / n + reg @ theta
            H = (Xb * (h * (1 - h))[:, None]).T @ Xb / n + reg
            step = np.linalg.solve(H, grad)
            theta -= step
            self.cost_history_.append(self.cost(Xb, y, theta))
            if np.abs(step).max() < self.tol:
                break
        self.n_iter_ = len(self.cost_history_) - 1
        self.theta_scaled_ = theta
        self.theta_ = self._unscale_theta(theta)           # [theta0, theta1, theta2]
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return _add_intercept(np.asarray(X, float)) @ self.theta_

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


class GDA(ClassifierMixin, BaseEstimator):
    """Generative model of Eq. 4: y ~ Bernoulli(phi), x|y ~ N(mu_y, Sigma).

    With a shared Sigma the posterior p(y=1|x) is a logistic function of a
    linear score, so the decision boundary p(y|x)=0.5 is a straight line.
    """

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        n = len(y)
        self.phi_ = y.mean()
        self.mu0_ = X[y == 0].mean(axis=0)
        self.mu1_ = X[y == 1].mean(axis=0)
        centred = X - np.where(y[:, None] == 1, self.mu1_, self.mu0_)
        self.sigma_ = centred.T @ centred / n
        inv = np.linalg.inv(self.sigma_)
        w = inv @ (self.mu1_ - self.mu0_)
        b = (-0.5 * self.mu1_ @ inv @ self.mu1_ + 0.5 * self.mu0_ @ inv @ self.mu0_
             + np.log(self.phi_ / (1 - self.phi_)))
        self.theta_ = np.concatenate([[b], w])
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return _add_intercept(np.asarray(X, float)) @ self.theta_

    def predict_proba(self, X):
        p = sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


class LinearSVM(ClassifierMixin, BaseEstimator, _Standardiser):
    """Soft-margin SVM:  min  lambda/2 ||w||^2 + 1/n sum max(0, 1 - y_i (w.x_i + b)).

    Solved with mini-batch Pegasos (Shalev-Shwartz et al.) and iterate
    averaging; labels are mapped to {-1, +1} internally.
    """

    def __init__(self, C: float = 1.0, epochs: int = 200, batch_size: int = 64, seed: int = 0):
        self.C, self.epochs, self.batch_size, self.seed = C, epochs, batch_size, seed

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, int)
        self._fit_scale(X)
        Xs = self._scale(X)
        ys = np.where(y == 1, 1.0, -1.0)
        n, d = Xs.shape
        lam = 1.0 / (self.C * n)
        rng = np.random.default_rng(self.seed)
        w, b = np.zeros(d), 0.0
        w_avg, b_avg, t, n_avg = np.zeros(d), 0.0, 0, 0
        self.objective_history_ = []
        for epoch in range(self.epochs):
            for idx in np.array_split(rng.permutation(n), max(1, n // self.batch_size)):
                t += 1
                eta = 1.0 / (lam * t)
                viol = ys[idx] * (Xs[idx] @ w + b) < 1
                gw = lam * w - (ys[idx][viol, None] * Xs[idx][viol]).sum(0) / len(idx)
                gb = -ys[idx][viol].sum() / len(idx)
                w -= eta * gw
                b -= eta * gb * 0.1                      # damped bias step
                norm = np.linalg.norm(w)                 # Pegasos projection
                if norm > 1 / np.sqrt(lam):
                    w *= (1 / np.sqrt(lam)) / norm
                if epoch >= self.epochs // 2:            # average 2nd half
                    w_avg += w
                    b_avg += b
                    n_avg += 1
            margins = ys * (Xs @ w + b)
            self.objective_history_.append(0.5 * lam * w @ w + np.maximum(0, 1 - margins).mean())
        w, b = w_avg / n_avg, b_avg / n_avg
        self.theta_scaled_ = np.concatenate([[b], w])
        self.theta_ = self._unscale_theta(self.theta_scaled_)
        self.classes_ = np.array([0, 1])
        return self

    def decision_function(self, X):
        return _add_intercept(np.asarray(X, float)) @ self.theta_

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


def make_models(seed: int = 0) -> dict:
    return {
        "Logistic Regression": LogisticRegressionNewton(),
        "GDA": GDA(),
        "Linear SVM": LinearSVM(seed=seed),
    }

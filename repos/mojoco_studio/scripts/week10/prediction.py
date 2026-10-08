"""Transparent NumPy implementations of the Chapter 7/9 prediction equations."""

from itertools import product

import numpy as np


def normalize(observations: np.ndarray) -> np.ndarray:
    """Map physical state (..., 3) = (q, qdot, remaining_fraction) to features.

    z=(q/1.5, qdot/4, 2*h-1), where q is rad, qdot rad/s and h in [0,1].
    Fixed scales of 1.5 rad and 4 rad/s put typical training states near [-1,1].
    They are feature-design choices (RL notes 9.5), not estimated statistics.
    No clipping is used: clipping would alias distinct out-of-range states.
    The physical angle is unwrapped; this exercise stays near the downward
    equilibrium and does not learn a periodic swing-up value function.
    """
    observations = np.asarray(observations, dtype=np.float64)
    if observations.shape[-1] != 3 or not np.isfinite(observations).all():
        raise ValueError("Expected finite (..., 3) observations")
    result = observations / np.array([1.5, 4.0, 1.0])
    result[..., 2] = 2 * observations[..., 2] - 1
    return result


class Features:
    """Fixed affine or Gaussian bases; both predictors are linear in weights."""

    def __init__(self, kind: str):
        if kind not in ("affine", "rbf"):
            raise ValueError("Feature kind must be affine or rbf")
        self.kind = kind
        self.centers = np.array(list(product(np.linspace(-1, 1, 7),
                                            np.linspace(-1, 1, 7),
                                            np.linspace(-1, 1, 5))))
        self.widths = np.array([1 / 3, 1 / 3, 0.5])
        self.size = 4 if kind == "affine" else len(self.centers) + 1

    def __call__(self, observations: np.ndarray) -> np.ndarray:
        """Construct x(s) with shape (..., d), including a constant basis.

        Affine: x=[1,z_q,z_v,z_h]/2, d=4. Scaling by 2 limits feature norm
        within the nominal cube but retains an affine function of raw state.
        RBF: b_i=exp(-sum_j((z_j-c_ij)/sigma_j)^2/2), a=[1,b_1,...,b_245],
        x=a/||a||_2, d=246. Centers form a 7x7x5 grid in normalized space;
        widths equal adjacent-center spacings. This implements the Gaussian
        construction of RL notes 9.5.5 with anisotropic coordinate scales.

        Unit-norm scaling is a project choice motivated by notes 9.6:
        a single update changes its own prediction by alpha*delta*||x||^2.
        It does not imply that equal alpha gives equal learning speed across
        bases. The bias prevents division by zero even outside the grid.
        Fixed nonlinear features still yield Vhat=x^T w linear in w (9.4).
        Correlated bases need not have full rank; we do not invert a Gram
        matrix or claim unique weights. Terminal values are masked by TD,
        rather than requiring the terminal feature vector to be zero.
        """
        z = normalize(observations)
        bias = np.ones((*z.shape[:-1], 1))
        if self.kind == "affine":
            return np.concatenate((bias, z), axis=-1) / 2
        distances = (z[..., None, :] - self.centers) / self.widths
        activations = np.exp(-0.5 * np.sum(distances**2, axis=-1))
        features = np.concatenate((bias, activations), axis=-1)
        return features / np.linalg.norm(features, axis=-1, keepdims=True)


def complete_returns(rewards: np.ndarray, gamma: float) -> np.ndarray:
    """Compute G_t=R_{t+1}+gamma*G_{t+1} backwards, with G_T=0.

    Input and output are (T,B): rewards[t,b] is R_{t+1} for world b;
    output[t,b] is the complete episodic return from S_t. All worlds truly
    terminate at T. This is NOT a time-limit truncation bootstrap. For our
    deterministic policy/simulator, each return is a numerical value reference
    at the observed state; stochastic tasks would need repeated rollouts.
    RL notes 7.1 and 9.3. Gamma must be in [0,1].
    """
    if not 0 <= gamma <= 1 or rewards.ndim != 2:
        raise ValueError("Expected gamma in [0,1] and rewards shaped (T,B)")
    result = np.empty_like(rewards, dtype=np.float64)
    running = np.zeros(rewards.shape[1])
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        result[t] = running
    return result


def n_step_target(rewards: np.ndarray, endpoint: np.ndarray, weights: np.ndarray,
                  gamma: float, terminal: bool) -> np.ndarray:
    """Build a delayed n-step target from k already observed transitions.

    rewards has shape (k,B), endpoint x(S_{t+k}) is (B,d), weights is (d,).
    G=sum_{i=0}^{k-1} gamma^i R_{t+i+1} + gamma^k x(S_{t+k})^T w if
    nonterminal; omit the entire second term at true termination. The last
    reward is always retained. A normal window has k=n; a flushed terminal
    window may have k<n. Evaluate the endpoint with CURRENT weights at backup
    time, not a cached prediction from collection time (RL notes 7.1, 9.4).
    This function returns numbers only: the update differentiates neither
    this bootstrap nor the environment. All B endpoints share terminal status
    because this exercise uses synchronized fixed-duration episodes.
    """
    if len(rewards) == 0 or not 0 <= gamma <= 1:
        raise ValueError("Need at least one reward and gamma in [0,1]")
    discounts = gamma ** np.arange(len(rewards))
    target = discounts @ rewards
    if not terminal:
        target = target + gamma ** len(rewards) * (endpoint @ weights)
    return target


def semi_gradient_update(weights: np.ndarray, features: np.ndarray,
                         target: np.ndarray, alpha: float) -> float:
    """Update w in place using a simultaneous mean of B semi-gradients.

    Vhat(S,w)=x(S)^T w, so grad_w Vhat=x. For X shaped (B,d), w shaped
    (d,), and frozen targets U shaped (B,), delta=U-Xw and
        w <- w + (alpha/B) X^T delta.
    All predictions read the same pre-update w. This batched extension equals
    textbook on-policy semi-gradient TD for B=1; it is not B sequential SGD
    steps. Increasing B reduces sampling noise but not the number of backups.
    Ignoring dU/dw makes this a semi-gradient when U bootstraps. Differentiating
    both endpoints would introduce -gamma*x(next), a different algorithm.
    RL notes 9.3-9.4. Return mean delta^2 BEFORE the update for diagnostics;
    it is not the true value error. A constant alpha gives no general exact
    convergence guarantee, even though our approximation is linear/on-policy.
    """
    if alpha <= 0 or not np.isfinite(alpha):
        raise ValueError("alpha must be finite and positive")
    error = target - features @ weights
    weights += alpha * (features.T @ error) / len(features)
    if not np.isfinite(weights).all():
        raise FloatingPointError("Value weights diverged; reduce alpha/check features")
    return float(np.mean(error**2))


class Predictor:
    """One fixed feature family and n-step horizon, with persistent weights."""

    def __init__(self, features: Features, n: int, alpha: float, gamma: float):
        if n < 1 or not 0 <= gamma <= 1 or not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("Need n>=1, gamma in [0,1], finite alpha>0")
        self.features, self.n, self.alpha, self.gamma = features, n, alpha, gamma
        self.weights = np.zeros(features.size)

    def learn(self, observations: np.ndarray, rewards: np.ndarray) -> float:
        """Apply the online delayed schedule to one batch of complete episodes.

        observations is (T+1,B,3), rewards (T,B). At step k, only rewards up
        to R_{k+1} are eligible, and tau=k-n+1 is updated if nonnegative.
        After T, flush through tau=T-1 without stepping the simulator. There
        are exactly T mean-batch updates, even if n>T. Current weights are
        read at EACH backup, including successive flush updates (notes 7.1).

        Trajectories are collected first for reuse by all compared predictors.
        The policy is fixed and independent of these weights, so consuming each
        transition once in this order gives the same updates as interleaving
        simulation and learning. Precomputing FIXED features leaks no future
        rewards into a target. This is on-policy prediction, not replay-based
        control. Return mean squared pre-update TD error over the T backups.
        """
        if (rewards.ndim != 2 or not len(rewards)
                or observations.shape != (len(rewards) + 1, rewards.shape[1], 3)):
            raise ValueError("Need observations (T+1,B,3), rewards (T,B), T>0")
        x = self.features(observations)
        horizon = len(rewards)
        errors = []
        for k in range(horizon + self.n - 1):
            tau = k - self.n + 1
            if tau < 0:
                continue
            end = min(tau + self.n, horizon)
            target = n_step_target(rewards[tau:end], x[end], self.weights,
                                   self.gamma, terminal=end == horizon)
            errors.append(semi_gradient_update(self.weights, x[tau], target, self.alpha))
        return float(np.mean(errors))


def value_rmse(predictions: np.ndarray, references: np.ndarray) -> float:
    """Return sqrt(mean((Vhat-G)^2)) over held-out visited states (RL 9.2).

    Both arrays have the same shape. Equal weighting of all sampled episode
    times/worlds approximates their empirical visitation distribution, not a
    uniform grid over continuous state space and not the squared TD residual.
    """
    if predictions.shape != references.shape:
        raise ValueError("Prediction/reference shapes must agree")
    return float(np.sqrt(np.mean((predictions - references)**2)))


def generalization_response(basis: Features, anchor: np.ndarray, queries: np.ndarray) -> np.ndarray:
    """Return x(s')^T x(s) for one anchor s=(3,) and queries s'=(...,3).

    From Delta w=alpha*delta*x(s), linearity gives the exact identity
    Delta Vhat(s')=alpha*delta*x(s')^T*x(s) (RL notes 9.4). Setting
    alpha*delta=1 exposes feature sharing independently of trained weights.
    Used by the generalization heatmap: affine bases spread an update globally;
    RBF overlap concentrates it nearby, with a global component from the bias.
    This is a single frozen-target update, not a multi-step learning guarantee.
    """
    return basis(queries) @ basis(anchor)

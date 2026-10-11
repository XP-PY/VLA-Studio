"""Action-block linear SARSA and accumulating eligibility traces (RL notes 10/12)."""

import numpy as np

from week10.prediction import Features

GOAL = 0.5


def state_features(states: np.ndarray) -> np.ndarray:
    """Encode (q,qdot,h), shape (B,3), using week 10's unit-norm RBF basis.

    Shift position by the fixed goal: z_q=(q-0.5)/1.5, z_v=qdot/4,
    z_h=2h-1. The 246 fixed features include a bias; output is (B,246).
    q is rad about world +y, qdot rad/s, h remaining episode fraction.
    Continuing tasks set h=0.5 forever, so there is no hidden time limit.
    Q(s,a)=W[a,:] dot x(s) uses separate action blocks (RL notes 10.1).
    Gaussian features are nonlinear in state but linear in the learned W.
    Fixed scaling does not clip or alias out-of-range observations.
    """
    shifted = np.array(states, dtype=float, copy=True)
    shifted[..., 0] -= GOAL
    return _BASIS(shifted)


_BASIS = Features("rbf")


def epsilon_greedy(values: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Sample one action per row of Q values (B,A), with uniform greedy ties.

    pi(a|s)=epsilon/A + (1-epsilon)*1[a in G(s)]/|G(s)|, where G is
    the exact maximizing set. Sample a uniform tie-break even at epsilon=0;
    otherwise zero-initialized actions would favor the first index. Exploration
    includes all actions, including greedy ones. No softmax or max-Q target is
    used: the selected successor action must be executed next (RL notes 10.1).
    """
    if not 0 <= epsilon <= 1 or values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Need finite (B,A) values and epsilon in [0,1]")
    ties = values == values.max(axis=1, keepdims=True)
    greedy = np.where(ties, rng.random(values.shape), -1).argmax(axis=1)
    explore = rng.random(len(values)) < epsilon
    random_actions = rng.integers(values.shape[1], size=len(values))
    return np.where(explore, random_actions, greedy)


class Sarsa:
    """Shared action weights with a separate trace for each on-policy world."""

    def __init__(self, actions=7, features=246, alpha=0.2, gamma=0.98,
                 lam=0.0, differential=False, beta=0.01):
        if (actions < 1 or features < 1 or not 0 <= gamma <= 1 or not 0 <= lam <= 1
                or not np.isfinite(alpha) or alpha <= 0 or not np.isfinite(beta) or beta <= 0):
            raise ValueError("Invalid SARSA dimensions or learning parameters")
        if differential and (gamma != 1 or lam != 0):
            raise ValueError("This differential exercise uses gamma=1, lambda=0")
        self.weights = np.zeros((actions, features))
        self.alpha, self.gamma, self.lam = alpha, gamma, lam
        self.differential, self.beta, self.rate = differential, beta, 0.0
        self.traces = None

    def reset_traces(self, batch: int) -> None:
        """Set z_{-1}=0 for B new episodes; retain W and the rate estimate.

        Trace storage is (B,A,D). Do not call this at logging boundaries of a
        continuing task. Terminal updates use their old episode traces before
        these are cleared (RL notes 12.7).
        """
        self.traces = np.zeros((batch, *self.weights.shape))

    def values(self, features: np.ndarray) -> np.ndarray:
        """Return Q(s,a)=sum_j X[s,j]*W[a,j], shape (B,A), from X=(B,D)."""
        return features @ self.weights.T

    def update(self, x, actions, rewards, next_x, next_actions, terminal=False) -> np.ndarray:
        """Take a simultaneous mean-batch semi-gradient SARSA(lambda) update.

        x,next_x are (B,D), actions,next_actions,rewards are (B,). Read ALL
        action values using pre-update W. For each world b:
          delta_b = R_{t+1} + gamma*Q(S',A') - Q(S,A),
          z_b <- gamma*lambda*z_b + one_hot(A) outer x(S),
          W <- W + alpha/B * sum_b delta_b*z_b.
        At a true terminal successor omit Q(S',A') entirely, keeping R and
        the trace update. All worlds share terminal status in this exercise.
        The caller selects A' BEFORE this update and carries it into the next
        interaction. Ordinary accumulating traces survive exploratory actions;
        these are not replacing or true-online traces (RL notes 12.7).
        lambda=0 reduces exactly to one-step semi-gradient SARSA (notes 10.1).
        The target is frozen: differentiating its W dependence would implement
        a different algorithm. Mean-batch updates equal the textbook for B=1,
        but are not B sequential single-transition updates.

        In differential mode use delta=R-rate+Q'-Q, gamma=1 and lambda=0;
        update rate <- rate+beta*mean(delta) with the SAME pre-update errors
        as W (notes 10.3). rate is reward per DECISION, not per second. There
        are no terminals or resets in this continuing variant; a collection
        cutoff is not a terminal transition. Differential values have an
        arbitrary additive offset. Constant step sizes and changing approximate
        control policies do not guarantee convergence or optimality.
        """
        if self.differential and terminal:
            raise ValueError("Differential SARSA has no terminal update")
        batch = len(x)
        if self.traces is None or self.traces.shape[0] != batch:
            raise ValueError("Initialize traces for this batch before updating")
        rows = np.arange(batch)
        current = np.sum(x * self.weights[actions], axis=1)
        successor = 0.0 if terminal else np.sum(next_x * self.weights[next_actions], axis=1)
        delta = rewards - (self.rate if self.differential else 0.0) + self.gamma * successor - current
        self.traces *= self.gamma * self.lam
        self.traces[rows, actions] += x
        self.weights += self.alpha * np.einsum("b,bad->ad", delta, self.traces) / batch
        if self.differential:
            self.rate += self.beta * float(delta.mean())
        if not np.isfinite(self.weights).all() or not np.isfinite(self.rate):
            raise FloatingPointError("SARSA diverged; inspect features and step sizes")
        return delta

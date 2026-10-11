"""Closed-form Gaussian scores make Chapter 13's actor updates inspectable."""

import numpy as np

from week10.prediction import complete_returns
from week11.control import GOAL, state_features

METHODS = ("reinforce", "reinforce_baseline", "actor_critic")


def actor_features(states):
    """Map (B,3)=(q [rad], qdot [rad/s], h) to (B,4) normalized features.

    x_mu=[1,(q-0.5)/1.5,qdot/4,2h-1]. Fixed scales match week 11; no
    clipping or running statistics. The fixed goal is 0.5 rad, h is remaining
    fraction of a true finite task. Continuing tasks fix h=0.5. A linear mean
    mu=theta_mu^T x can learn position/velocity feedback and a holding offset;
    it is not initialized from PD gains or gravity (RL notes 13.7).
    """
    states = np.asarray(states, dtype=float)
    if states.ndim != 2 or states.shape[1] != 3 or not np.isfinite(states).all():
        raise ValueError("Expected finite (B,3) states")
    return np.column_stack((np.ones(len(states)), (states[:, 0]-GOAL)/1.5,
                            states[:, 1]/4, 2*states[:, 2]-1))


def gaussian_log_probability(raw, mean, log_std):
    """Return log density of the UNCLIPPED normalized Gaussian sample u.

    log p(u|s)=-log(sigma)-log(2*pi)/2-(u-mu)^2/(2*sigma^2),
    sigma=exp(log_std). raw and mean have equal shape; log_std is scalar.
    Density is not the probability of an exact real number. This helper is
    used for score finite-difference checks (RL notes 13.7). Motor saturation
    produces probability atoms at torque limits; its output does NOT have
    this ordinary Gaussian density.
    """
    return -log_std - .5*np.log(2*np.pi) - .5*((raw-mean)/np.exp(log_std))**2


def gaussian_score(features, raw, mean, log_std):
    """Return grad_theta log p(u|s), shape (B,D+1), holding raw u fixed.

    For mu=theta_mu^T x and eta=log(sigma), differentiation yields
      dlogp/dtheta_mu = (u-mu)/sigma^2 * x,
      dlogp/deta = (u-mu)^2/sigma^2 - 1.
    Features are (B,D), raw/mean are (B,); eta is one learned state-independent
    parameter (x_sigma=1 in RL notes 13.7). Do not differentiate the sampling
    expression u=mu+sigma*z: that would not be this score estimator.

    Physics receives tau=3*clip(u,-1,1) N m. We regard clipping as a fixed,
    parameter-independent actuator inside the transition kernel, and score
    the latent u. The likelihood-ratio identity remains valid for this
    augmented action MDP. Scoring clip(u) as if it were Gaussian, or zeroing
    the score at saturation, would be wrong. This clipping convention extends
    the book's unbounded Gaussian example; it is not a Gaussian density over
    executed bounded torques and requires no tanh change-of-variables term.
    """
    residual = (raw-mean)/np.exp(log_std)
    return np.column_stack((features*(residual/np.exp(log_std))[:, None], residual**2-1))


def episodic_gradient(scores, advantages, gamma):
    """Return one frozen-policy gradient estimate per episode, shape (B,P).

    Input scores (T,B,P) and advantages (T,B) correspond to the SAME policy
    that generated that batch. g_b=sum_t gamma^t*A_{t,b}*score_{t,b} estimates
    grad J for J=E[sum_t gamma^t R_{t+1}]. The outer gamma^t discounts visits
    relative to episode start; returns/advantages already discount future
    rewards relative to t. Do not silently omit either factor (RL 13.2-13.4).
    A=G for REINFORCE or G-V_old(S) for its frozen state baseline. We average
    these gradients across B episodes, not across T*B transitions. Averaging
    over time as well would rescale the step size for this fixed horizon.
    """
    if not 0 <= gamma <= 1 or scores.shape[:2] != advantages.shape:
        raise ValueError("Need gamma in [0,1], matching (T,B) dimensions")
    return np.einsum("t,tb,tbp->bp", gamma**np.arange(len(scores)), advantages, scores)


def mean_variance(gradients):
    """Estimate trace of episode-gradient covariance: sum_j Var(g_j), ddof=1.

    gradients is (B,P), B>=2 (return NaN if only one episode is available).
    This sample dispersion describes variation of
    individual episode gradients at one fixed policy, not standard error of
    the batch mean. Paired baseline/no-baseline diagnostics use the exact same
    returns and scores; neither value is a theorem about global performance.
    """
    if len(gradients) < 2:
        return float("nan")
    return float(np.var(gradients, axis=0, ddof=1).sum())


class GaussianActor:
    """Linear normalized-torque mean plus learned scalar log standard deviation."""

    def __init__(self, alpha=.02, std=.5, max_gradient=10.):
        if (not np.isfinite(alpha) or alpha <= 0 or not .1 <= std <= 2
                or not np.isfinite(max_gradient) or max_gradient <= 0):
            raise ValueError("Need alpha>0, std in [0.1,2], gradient limit>0")
        self.theta = np.zeros(5)
        self.theta[-1] = np.log(std)
        self.alpha, self.max_gradient = alpha, max_gradient

    def sample(self, features, rng):
        """Sample u=mu+sigma*z and its pre-update score, with z~N(0,1).

        features (B,4) gives mu (B,), scalar sigma in normalized torque units.
        Return raw u, scores (B,5), and mu; the environment alone bounds torque.
        Sampling is performed numerically, with no pathwise gradient or
        differentiation of the simulator. Keep raw actions for reproducibility.
        """
        mean = features @ self.theta[:-1]
        raw = mean + np.exp(self.theta[-1])*rng.normal(size=len(features))
        return raw, gaussian_score(features, raw, mean, self.theta[-1]), mean

    def ascend(self, gradient):
        """Apply theta <- theta + alpha*g, with explicit numerical safeguards.

        g is the mean gradient (5,). Project its Euclidean norm to at most 10
        by default, preserving direction but changing stochastic step size.
        Then project eta to [log(.1),log(2)] to prevent collapsed or explosive
        exploration. These project choices are not book convergence results;
        the raw estimator is the score-based gradient, the safeguarded update
        is not an unbiased estimator of an unconstrained parameter increment.
        Return the raw norm and whether norm clipping was used for logging.
        """
        gradient = np.asarray(gradient)
        if gradient.shape != self.theta.shape or not np.isfinite(gradient).all():
            raise FloatingPointError("Invalid Gaussian policy gradient")
        norm = float(np.linalg.norm(gradient))
        self.theta += self.alpha*gradient*min(1., self.max_gradient/max(norm, 1e-12))
        self.theta[-1] = np.clip(self.theta[-1], np.log(.1), np.log(2.))
        if not np.isfinite(self.theta).all():
            raise FloatingPointError("Nonfinite actor parameters")
        return norm, norm > self.max_gradient


class Learner:
    """Separate actor/critic parameters; batch Monte Carlo or online one-step AC."""

    def __init__(self, method, actor_alpha=.02, critic_alpha=.2, gamma=.98):
        if (method not in METHODS or not 0 <= gamma <= 1
                or not np.isfinite(critic_alpha) or critic_alpha <= 0):
            raise ValueError("Invalid algorithm, gamma, or critic step size")
        self.method, self.gamma, self.critic_alpha = method, gamma, critic_alpha
        self.actor = GaussianActor(actor_alpha)
        self.weights = np.zeros(246)
        self.rate = 0.

    def value(self, features):
        """Return Vhat=Xw, with X=(B,246), w=(246,), value output (B,)."""
        return features @ self.weights

    def monte_carlo(self, scores, value_features, rewards):
        """Update once after full episodes, freezing scores AND baseline first.

        scores (T,B,5) were computed at unchanged collection-policy parameters;
        value_features is (T,B,246), rewards[t,b]=R_{t+1}. Compute G backwards
        with G_T=0. For baseline REINFORCE, calculate V_old=Xw and A=G-V_old
        BEFORE fitting w to this batch. Otherwise b=0 and A=G. Update actor
        by alpha_theta/B*sum_b sum_t gamma^t*A*score. Baseline learning uses
        w <- w + alpha_w/(T*B)*sum A*x: a mean stochastic gradient for squared
        MC value error. Return paired empirical gradient-variance diagnostics.

        The action-independent baseline cancels in expectation because
        E[score|s]=grad integral pi(u|s)du=0 (RL notes 13.4). Fitting the current
        batch first could couple the baseline to its sampled actions and invalidate
        that simple argument. We use neither return/advantage normalization nor
        repeated policy epochs. MC targets only center returns, so this baseline
        algorithm is not actor-critic in the book's bootstrapping terminology.
        """
        if self.method == "actor_critic":
            raise ValueError("Actor-critic must update each transition")
        returns = complete_returns(rewards, self.gamma)
        baseline = np.einsum("tbd,d->tb", value_features, self.weights)
        advantage = returns-baseline if self.method == "reinforce_baseline" else returns
        gradients = episodic_gradient(scores, advantage, self.gamma)
        norm, clipped = self.actor.ascend(gradients.mean(axis=0))
        if self.method == "reinforce_baseline":
            self.weights += self.critic_alpha*np.einsum("tb,tbd->d", advantage, value_features)/advantage.size
        if not np.isfinite(self.weights).all():
            raise FloatingPointError("Nonfinite Monte Carlo baseline weights")
        return dict(gradient_norm=norm, gradient_clipped=float(clipped),
                    gradient_variance_plain=mean_variance(episodic_gradient(scores, returns, self.gamma)),
                    gradient_variance_baseline=mean_variance(gradients))

    def transition(self, scores, features, rewards, next_features, time_index, terminal=False,
                   continuing=False, rate_alpha=.01):
        """One-step actor-critic, using one shared pre-update TD error for both.

        scores=(B,5) is the current sampled-action score; X,X'=(B,246).
        Episodic: delta=R+gamma*V_old(S')-V_old(S); mask V(S') at a TRUE
        terminal successor. Then w+=alpha_w/B*X^T delta and
        theta+=alpha_theta/B*sum_b gamma^t*delta_b*score_b (RL 13.5).
        Both predictions and the policy score use pre-update parameters;
        neither critic target nor advantage is differentiated by the actor.
        An approximate successor value can bias the actor update even though
        subtracting an action-independent baseline does not (notes 13.5).

        Continuing: delta=R-rate+V_old(S')-V_old(S), no gamma^t and no
        termination. Also rate+=rate_alpha*mean(delta), reusing the SAME error
        before changing any parameter (notes 13.6, zero trace parameters).
        Rate is reward per decision; no reset is allowed at collection cutoffs.
        Finite batches and constant step sizes do not prove optimality or
        ergodicity. Return diagnostics; a norm bound is applied by ascend().
        """
        if self.method != "actor_critic" or continuing and terminal:
            raise ValueError("Invalid actor-critic update mode")
        current = self.value(features)
        successor = 0. if terminal else self.value(next_features)
        discount = 1. if continuing else self.gamma
        delta = rewards-(self.rate if continuing else 0.)+discount*successor-current
        factor = 1. if continuing else self.gamma**time_index
        norm, clipped = self.actor.ascend(np.mean(factor*delta[:, None]*scores, axis=0))
        self.weights += self.critic_alpha*(features.T @ delta)/len(features)
        if continuing:
            self.rate += rate_alpha*float(delta.mean())
        if not np.isfinite(self.weights).all() or not np.isfinite(self.rate):
            raise FloatingPointError("Nonfinite critic or reward-rate estimate")
        return dict(gradient_norm=norm, gradient_clipped=float(clipped), td_mse=float(np.mean(delta**2)))

"""Chapter 13 score/gradient arithmetic and continuous MuJoCo contracts."""

import math
import os
from pathlib import Path
import sys
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts"))
from week12.policy_gradient import (GaussianActor, Learner, actor_features, episodic_gradient,
                                    gaussian_log_probability, gaussian_score)
from week12.experiment import ContinuousPendulum, train_batch


class GradientTests(unittest.TestCase):
    def test_book_gaussian_score(self):
        score = gaussian_score(np.ones((1, 1)), np.array([3.]), np.array([0.]), np.log(2.))
        np.testing.assert_allclose(score, [[.75, 1.25]])

    def test_score_matches_finite_differences_holding_action_fixed(self):
        phi = np.array([[1., -.3, .2, .4]])
        theta = np.array([.1, .4, -.2, .3, np.log(.7)])
        raw = np.array([1.4])  # Saturated sample must still get its raw score.
        analytic = gaussian_score(phi, raw, phi @ theta[:-1], theta[-1])[0]
        numeric = []
        for i in range(5):
            plus, minus = theta.copy(), theta.copy()
            plus[i] += 1e-6
            minus[i] -= 1e-6
            a = gaussian_log_probability(raw, phi@plus[:-1], plus[-1])
            b = gaussian_log_probability(raw, phi@minus[:-1], minus[-1])
            numeric.append(((a-b)/2e-6).item())
        np.testing.assert_allclose(analytic, numeric, rtol=1e-7, atol=1e-8)

    def test_raw_score_for_clipped_action_objective(self):
        # For J(mu)=E[clip(U,-1,1)], dJ/dmu=P(-1<U<1). Numerically
        # integrate the score identity, including both saturated Gaussian tails.
        raw = np.linspace(-7., 7., 100001)
        mu, sigma = .2, .7
        density = np.exp(gaussian_log_probability(raw, mu, np.log(sigma)))
        score = (raw-mu)/sigma**2
        actual = np.trapezoid(np.clip(raw, -1, 1)*score*density, raw)
        cdf = lambda x: .5*(1+math.erf((x-mu)/(sigma*math.sqrt(2))))
        self.assertAlmostEqual(actual, cdf(1)-cdf(-1), places=7)
        wrong = np.trapezoid(np.clip(raw, -1, 1)*(np.clip(raw,-1,1)-mu)/sigma**2*density, raw)
        self.assertGreater(abs(wrong-actual), .05)

    def test_outer_discount_and_episode_averaging(self):
        score = np.array([[[1.]], [[2.]], [[3.]]])
        advantage = np.array([[1.], [2.], [3.]])
        np.testing.assert_allclose(episodic_gradient(score, advantage, .5), [[5.25]])

    def test_baseline_is_frozen_before_current_batch_fit(self):
        learner = Learner("reinforce_baseline", actor_alpha=.1, critic_alpha=.5, gamma=.5)
        learner.weights[0] = 2.
        x = np.zeros((2, 2, 246))
        x[:, :, 0] = 1
        old = learner.actor.theta.copy()
        learner.monte_carlo(np.ones((2, 2, 5)), x, np.array([[1., 2.], [3., 4.]]))
        np.testing.assert_allclose(learner.actor.theta-old, np.full(5, .2))
        self.assertAlmostEqual(learner.weights[0], 2.6875)

    def test_actor_critic_preupdate_values_and_outer_discount(self):
        learner = Learner("actor_critic", actor_alpha=.1, critic_alpha=.1, gamma=.9)
        learner.weights[0] = 2
        x = np.zeros((1, 246))
        x[0, 0] = 1
        score = np.array([[1., 0., 0., 0., 0.]])
        learner.transition(score, x, np.array([1.]), x*1.5, 2)
        self.assertAlmostEqual(learner.weights[0], 2.17)
        self.assertAlmostEqual(learner.actor.theta[0], .1*.9**2*1.7)

    def test_terminal_keeps_reward_and_omits_successor(self):
        learner = Learner("actor_critic", actor_alpha=.1, critic_alpha=.1)
        x = np.zeros((1, 246)); x[0, 0] = 1
        learner.transition(np.array([[1.,0,0,0,0]]), x, np.array([2.]),
                           np.full_like(x, np.nan), 0, terminal=True)
        self.assertAlmostEqual(learner.weights[0], .2)
        self.assertAlmostEqual(learner.actor.theta[0], .2)

    def test_continuing_uses_same_error_and_no_outer_discount(self):
        learner = Learner("actor_critic", actor_alpha=.1, critic_alpha=.1, gamma=.5)
        learner.weights[0], learner.rate = 3., 2.
        x = np.zeros((1,246)); x[0,0]=1
        learner.transition(np.array([[1.,0,0,0,0]]), x, np.array([4.]), x*(2/3),
                           100, continuing=True, rate_alpha=.05)
        self.assertAlmostEqual(learner.rate, 2.05)
        self.assertAlmostEqual(learner.weights[0], 3.1)
        self.assertAlmostEqual(learner.actor.theta[0], .1)
        with self.assertRaises(ValueError):
            learner.transition(np.zeros((1,5)), x, np.zeros(1), x, 0, terminal=True, continuing=True)

    def test_actor_projection_is_explicit(self):
        actor = GaussianActor(alpha=1.)
        norm, clipped = actor.ascend(np.array([0.,0,0,0,100]))
        self.assertEqual(norm, 100)
        self.assertTrue(clipped)
        self.assertAlmostEqual(np.exp(actor.theta[-1]), 2.)

    def test_mc_policy_frozen_during_collection_and_online_ac_timing(self):
        env = ContinuousPendulum(["torque"]*6, duration=.08, backend="native")
        names = ("reinforce", "reinforce_baseline", "actor_critic")
        learners = {name: Learner(name) for name in names}
        snapshots = {name: [] for name in names}
        updates = {}

        def observer(name, original):
            def sample(features, rng):
                snapshots[name].append(learners[name].actor.theta.copy())
                return original(features, rng)
            return sample

        with ExitStack() as stack:
            for name, learner in learners.items():
                stack.enter_context(patch.object(learner.actor, "sample",
                                                  side_effect=observer(name, learner.actor.sample)))
                updates[name] = stack.enter_context(patch.object(learner.actor, "ascend", wraps=learner.actor.ascend))
            train_batch(env, learners, np.array([[.1, -.2],[.6, .1]]),
                        {name: np.random.default_rng(i) for i,name in enumerate(names)})
        for name in names[:2]:
            np.testing.assert_equal(snapshots[name][0], snapshots[name][1])
            self.assertEqual(updates[name].call_count, 1)
        self.assertEqual(updates["actor_critic"].call_count, 2)
        self.assertGreater(np.linalg.norm(snapshots["actor_critic"][1]-snapshots["actor_critic"][0]), 0.)


class ContinuousRobotTests(unittest.TestCase):
    def test_clipping_and_continuing_state(self):
        env = ContinuousPendulum(["torque", "torque"], duration=.04, backend="native", continuing=True)
        env.reset(np.zeros((2,2)))
        for _ in range(2):
            state, _, terminal, info = env.step(np.array([2., -2.]))
            self.assertFalse(terminal)
            np.testing.assert_equal(state[:,2], .5)
            np.testing.assert_allclose(info["torque_integral"], [.12, -.12])
            np.testing.assert_allclose(info["saturated_time"], [.04,.04])
        with self.assertRaises(ValueError):
            env.step(np.array([np.nan, 0]))

    @unittest.skipUnless(os.getenv("WEEK12_GPU_TESTS")=="1", "set WEEK12_GPU_TESTS=1")
    def test_cpu_gpu_continuous_rollout_parity(self):
        a = ContinuousPendulum(["torque", "pd_gravity"], duration=.2, backend="native")
        b = ContinuousPendulum(["torque", "pd_gravity"], duration=.2)
        initial = np.array([[.3, -.2], [.1, .2]])
        a.reset(initial); b.reset(initial)
        for u in (.2, 1.5, -.4, -2., .7):
            ra = a.step(np.array([u, 0.])); rb = b.step(np.array([u, 0.]))
            np.testing.assert_allclose(ra[0], rb[0], atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(ra[1], rb[1], atol=1e-5, rtol=3e-4)
            self.assertEqual(ra[2], rb[2])


if __name__=="__main__":
    unittest.main()

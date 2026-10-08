"""Equation-level checks plus native/GPU environment contract checks."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from week10.pendulum import Pendulum, policy, reward, tip_kinematics
from week10.prediction import (Features, Predictor, complete_returns, normalize,
                               n_step_target, semi_gradient_update)


class IdentityFeatures:
    size = 3

    def __call__(self, states):
        return states


class PredictionTests(unittest.TestCase):
    def test_book_return_example_and_terminal_reward(self):
        rewards = np.array([[1.0], [2.0], [3.0]])
        np.testing.assert_allclose(n_step_target(rewards, np.ones((1, 1)),
                                                np.array([4.0]), .9, False), [8.146])
        # A terminal feature may be nonzero: no continuation, last reward stays.
        np.testing.assert_allclose(n_step_target(rewards, np.ones((1, 1)),
                                                np.array([1000.0]), .9, True), [5.23])
        np.testing.assert_allclose(complete_returns(rewards, .9)[:, 0], [5.23, 4.7, 3])

    def test_simultaneous_batch_update(self):
        weights = np.array([.5, -1.0])
        semi_gradient_update(weights, np.array([[1., 2.], [2., -1.]]), np.array([3., 4.]), .2)
        np.testing.assert_allclose(weights, [1.35, -.3])

    def test_delayed_schedule_and_short_episode_flush(self):
        states = np.vstack((np.eye(3), np.zeros(3)))[:, None, :]
        rewards = np.array([[1.], [2.], [3.]])
        for n, expected in ((1, [1, 2, 3]), (2, [2.8, 4.7, 3]),
                            (4, [5.23, 4.7, 3])):
            learner = Predictor(IdentityFeatures(), n, 1., .9)
            with patch("week10.prediction.semi_gradient_update", wraps=semi_gradient_update) as update:
                learner.learn(states, rewards)
                self.assertEqual(update.call_count, 3)
            np.testing.assert_allclose(learner.weights, expected)

    def test_bootstrap_uses_current_weights(self):
        states = np.tile([1., 0., 0.], (5, 1, 1))
        learner = Predictor(IdentityFeatures(), 2, 1., .9)
        targets = []

        def capture(w, x, target, alpha):
            targets.append(target.copy())
            return semi_gradient_update(w, x, target, alpha)

        with patch("week10.prediction.semi_gradient_update", side_effect=capture):
            learner.learn(states, np.arange(1., 5.)[:, None])
        np.testing.assert_allclose(np.array(targets)[:, 0], [2.8, 6.968, 6.6, 4.])

    def test_features_normalization_and_generalization(self):
        np.testing.assert_allclose(normalize(np.array([1.5, 4., .5])), [1., 1., 0.])
        self.assertGreater(normalize(np.array([3., 0., .5]))[0], 1.)  # no clipping
        states = np.array([[0., 0., .5], [.01, 0., .5], [1.4, 3., .5]])
        basis = Features("rbf")
        x = basis(states)
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.)
        weights = np.zeros(basis.size)
        semi_gradient_update(weights, x[:1], np.array([2.]), .1)
        np.testing.assert_allclose(x @ weights, .2 * (x @ x[0]))
        self.assertGreater(x[1] @ weights, x[2] @ weights)
        self.assertFalse(np.allclose(basis([[0, 0, .2]]), basis([[0, 0, .8]])))


class EnvironmentTests(unittest.TestCase):
    def test_timing_terminal_and_reset(self):
        env = Pendulum(2, 3, "native")
        initial = np.array([[.3, .2], [-.4, .1]])
        states = env.reset(initial)
        expected = reward(states, policy(states))
        successor, received, terminated = env.step()
        np.testing.assert_allclose(received, expected)
        np.testing.assert_allclose(successor[:, 2], 2 / 3)
        self.assertFalse(terminated)
        env.step()
        end, last_reward, terminated = env.step()
        self.assertTrue(terminated)
        np.testing.assert_equal(end[:, 2], 0)
        self.assertTrue(np.all(last_reward < 0))
        with self.assertRaises(RuntimeError):
            env.step()
        np.testing.assert_allclose(env.reset(initial), states)
        self.assertEqual(env.data[0].time, 0)

    def test_tip_jacobian_against_mujoco(self):
        env = Pendulum(1, 1, "native")
        state = env.reset(np.array([[.7, -.8]]))
        position, velocity = tip_kinematics(state)
        data = env.data[0]
        site = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, "tip")
        jacobian = np.zeros((3, env.model.nv))
        mujoco.mj_jacSite(env.model, data, jacobian, None, site)
        np.testing.assert_allclose(position[0], data.site_xpos[site], atol=1e-12)
        np.testing.assert_allclose(velocity[0], jacobian @ data.qvel, atol=1e-12)

    @unittest.skipUnless(os.getenv("WEEK10_GPU_TESTS") == "1", "set WEEK10_GPU_TESTS=1 for CUDA")
    def test_cpu_gpu_rollout_and_reset_parity(self):
        initial = np.array([[.4, -.2], [-.7, .8]])
        cpu = Pendulum(2, 20, "native")
        gpu = Pendulum(2, 20, "mjlab")
        states, rewards = cpu.rollout(initial)
        gpu_states, gpu_rewards = gpu.rollout(initial)
        np.testing.assert_allclose(gpu_states, states, atol=2e-4, rtol=2e-4)
        np.testing.assert_allclose(gpu_rewards, rewards, atol=2e-5, rtol=2e-4)
        again, again_rewards = gpu.rollout(initial)
        np.testing.assert_allclose(again, gpu_states, atol=1e-6)
        np.testing.assert_allclose(again_rewards, gpu_rewards, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

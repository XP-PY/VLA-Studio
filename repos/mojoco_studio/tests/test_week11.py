"""Hand-computable RL updates, control identities, and simulator contracts."""

import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from week11.control import Sarsa, epsilon_greedy
from week11.experiment import train_episode
from week11.robot import ControlPendulum, cartesian_target, gravity_torque, motor_torque


class LearningTests(unittest.TestCase):
    def test_sampled_sarsa_and_terminal_mask(self):
        learner = Sarsa(actions=2, features=1, alpha=.1, gamma=.9)
        learner.weights[:, 0] = [1., 5.]
        learner.reset_traces(1)
        # Sampled A'=0 has Q=1; max would be 5 and produce a different target.
        error = learner.update(np.ones((1, 1)), np.array([0]), np.array([2.]),
                               np.ones((1, 1)), np.array([0]))
        np.testing.assert_allclose(error, [1.9])
        np.testing.assert_allclose(learner.weights[:, 0], [1.19, 5.])
        learner.reset_traces(1)
        learner.update(np.ones((1, 1)), np.array([1]), np.array([3.]),
                       np.array([[1000.]]), np.array([1]), True)
        self.assertAlmostEqual(learner.weights[1, 0], 4.8)

    def test_delayed_credit_accumulates_and_resets(self):
        learner = Sarsa(actions=1, features=3, alpha=.1, gamma=1, lam=.8)
        learner.reset_traces(1)
        for i in range(3):
            learner.update(np.eye(3)[i:i+1], np.array([0]), np.array([float(i == 2)]),
                           np.zeros((1, 3)), np.array([0]), terminal=i == 2)
        np.testing.assert_allclose(learner.weights[0], [.064, .08, .1])
        learner.reset_traces(1)
        np.testing.assert_equal(learner.traces, 0)

    def test_lambda_zero_and_mean_batch(self):
        learner = Sarsa(actions=2, features=1, alpha=.2, gamma=0., lam=0.)
        learner.reset_traces(2)
        learner.traces[:] = 100  # must disappear at lambda=0
        learner.update(np.array([[1.], [2.]]), np.array([0, 1]), np.array([2., 3.]),
                       np.zeros((2, 1)), np.array([0, 0]))
        np.testing.assert_allclose(learner.weights[:, 0], [.2, .6])

    def test_differential_same_preupdate_error(self):
        learner = Sarsa(actions=2, features=1, alpha=.1, gamma=1, differential=True, beta=.05)
        learner.weights[:, 0], learner.rate = [3., 2.], 2.
        learner.reset_traces(1)
        delta = learner.update(np.ones((1, 1)), np.array([0]), np.array([4.]),
                               np.ones((1, 1)), np.array([1]))
        np.testing.assert_allclose(delta, [1.])
        self.assertAlmostEqual(learner.rate, 2.05)
        self.assertAlmostEqual(learner.weights[0, 0], 3.1)
        with self.assertRaises(ValueError):
            learner.update(np.ones((1, 1)), np.array([0]), np.array([0.]),
                           np.ones((1, 1)), np.array([0]), terminal=True)

    def test_uniform_ties_and_epsilon_probabilities(self):
        values = np.tile([2., 2., 0.], (30000, 1))
        selected = epsilon_greedy(values, .3, np.random.default_rng(77))
        np.testing.assert_allclose(np.bincount(selected, minlength=3) / len(selected),
                                   [.45, .45, .1], atol=.012)

    def test_runner_executes_preupdate_successor_action(self):
        class TinyEnv:
            horizon = 2
            seen = []

            def reset(self, _):
                return np.zeros((1, 3))

            def step(self, actions):
                self.seen.append(int(actions[0]))
                return np.zeros((1, 3)), -np.ones(1), len(self.seen) == 2, {}

        learner = Sarsa(actions=2, features=1, alpha=2, gamma=0)
        learner.weights[:, 0] = [0., .1]  # pre-update greedy choice is action 1
        env = TinyEnv()
        with patch("week11.experiment.state_features", return_value=np.ones((1, 1))):
            train_episode(env, {"one": learner}, np.zeros((1, 2)),
                          {"one": np.random.default_rng(1)}, 0.)
        self.assertEqual(env.seen, [1, 1])


class RobotTests(unittest.TestCase):
    def test_gravity_against_mujoco_bias(self):
        env = ControlPendulum(["pd_gravity"], backend="native")
        for q in (-.8, 0., .5, 1.1):
            env.reset(np.array([[q, 0.]]))
            self.assertAlmostEqual(gravity_torque(q), env.data[0].qfrc_bias[0], places=10)

    def test_cartesian_inverse_and_singularity(self):
        q = np.array([.2])
        delta = np.array([1e-4])
        result = cartesian_target(q, delta)
        j = -.6 * np.cos(q)
        np.testing.assert_allclose(result - q, j * delta / (j*j + .05**2))
        self.assertTrue(np.isfinite(cartesian_target(np.array([np.pi / 2]), np.array([.08]))).all())
        self.assertLessEqual(abs(result[0] - q[0]), .2)

    def test_torque_interface_does_not_add_gravity(self):
        torque, _ = motor_torque(np.array([.5, .5]), np.zeros(2),
                                 np.array([1., .5]), np.array([0, 4]))
        np.testing.assert_allclose(torque, [1., gravity_torque(.5)])
        capped, _ = motor_torque(np.array([-.5]), np.zeros(1), np.array([1.]), np.array([1]))
        self.assertEqual(capped[0], 3.)

    def test_goal_equilibrium_reward_units_and_horizon(self):
        env = ControlPendulum(["pd_gravity"], duration=.08, backend="native")
        env.reset(np.array([[.5, 0.]]))
        state, reward, terminal, info = env.step(np.array([0]))
        np.testing.assert_allclose(state[0, :2], [.5, 0.], atol=1e-12)
        self.assertAlmostEqual(reward[0], -.04 * .005 * gravity_torque(.5)**2)
        self.assertAlmostEqual(info["work_abs"][0], 0.)
        self.assertFalse(terminal)
        self.assertTrue(env.step(np.array([0]))[2])
        with self.assertRaises(RuntimeError):
            env.step(np.array([0]))

    def test_continuing_cutoff_is_not_terminal(self):
        env = ControlPendulum(["joint"], duration=.04, backend="native", continuing=True)
        env.reset(np.array([[.5, 0.]]))
        for _ in range(3):
            state, _, terminal, _ = env.step(np.array([3]))
            self.assertFalse(terminal)
            self.assertEqual(state[0, 2], .5)
        self.assertAlmostEqual(env.data[0].time, .12)

    @unittest.skipUnless(os.getenv("WEEK11_GPU_TESTS") == "1", "set WEEK11_GPU_TESTS=1")
    def test_gpu_cpu_all_interfaces_and_kick(self):
        names = ["joint", "cartesian", "torque", "pd", "pd_gravity"]
        cpu = ControlPendulum(names, duration=.4, backend="native")
        gpu = ControlPendulum(names, duration=.4)
        initial = np.tile([.2, -.3], (len(names), 1))
        cpu.reset(initial)
        gpu.reset(initial)
        for step in range(10):
            if step == 5:
                cpu.kick(np.full(5, .7))
                gpu.kick(np.full(5, .7))
            action = np.array([3, 2, 4, 0, 0])
            a = cpu.step(action, np.full(5, .2))
            b = gpu.step(action, np.full(5, .2))
            np.testing.assert_allclose(a[0], b[0], atol=3e-4, rtol=3e-4)
            np.testing.assert_allclose(a[1], b[1], atol=1e-5, rtol=3e-4)
            self.assertEqual(a[2], b[2])


if __name__ == "__main__":
    unittest.main()

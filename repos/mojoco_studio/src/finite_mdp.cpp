/*
  Week 4 reinforcement-learning exercise: policy evaluation in a finite MDP.

  A five-state line world has terminal states at both ends. In each nonterminal
  state the fixed policy selects LEFT and RIGHT with equal probability. An
  action moves in its intended direction with probability 0.8 and leaves the
  agent in place with probability 0.2. Every nonterminal transition costs -1.

  The state values are calculated with synchronous Bellman expectation backups:

    V_pi(s) = sum_a pi(a|s) sum_{s',r} p(s',r|s,a)
              [r + gamma V_pi(s')].
*/

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kStateCount = 5;    // Number of states: 5 (indices 0–4); 
                                  // the two ends are terminal states (0 and 4), 
                                  // and the three intermediate ones are non-terminal states (1, 2, 3).
constexpr int kActionCount = 2;   // Number of actions: 2 (LEFT and RIGHT)
constexpr double kDiscount = 0.9; // Discount factor
constexpr double kConvergenceTolerance = 1e-12; // Convergence tolerance
constexpr int kMaximumIterations = 100000;      // Maximum number of iterations

enum class Action : int {
  kLeft = 0,
  kRight = 1,
};

struct Outcome {
  int next_state;
  double probability;
  double reward;
};

// An array of length kStateCount storing the value of each state.
using ValueTable = std::array<double, kStateCount>;

// A kStateCount×kActionCount two-dimensional array, 
// where `policy[state][action]` represents the probability of taking action `a` in state `s`. 
// Action probabilities for terminal states are unused but initialized to 0.
using Policy = std::array<std::array<double, kActionCount>, kStateCount>;

struct IterationRecord {
  int iteration;
  ValueTable values;
  double maximum_update;
};

bool is_terminal(int state) {
  return state == 0 || state == kStateCount - 1;
}

// Return p(s',r|s,a) as a short list of outcomes. Terminal-state values are
// fixed at zero, so their self-transition carries zero reward.
std::vector<Outcome> transition_outcomes(int state, Action action) {
  if (is_terminal(state)) {
    return {{state, 1.0, 0.0}};
  }

  const int direction = action == Action::kLeft ? -1 : 1;
  const int intended_state = state + direction;
  return {
      {intended_state, 0.8, -1.0},
      {state, 0.2, -1.0},
  };
}

// The fixed stochastic policy chooses both actions uniformly in nonterminal
// states. Terminal states have no decision, so their action probabilities are 0.
Policy make_fixed_policy() {
  Policy policy{};
  for (int state = 0; state < kStateCount; ++state) {
    if (!is_terminal(state)) {
      policy[state][static_cast<int>(Action::kLeft)] = 0.5;
      policy[state][static_cast<int>(Action::kRight)] = 0.5;
    }
  }
  return policy;
}

// Apply the Bellman expectation operator to one state using the old value
// table. This separation makes the mathematical equation visible in code.
double bellman_expectation_backup(int state,
                                  const ValueTable& values,
                                  const Policy& policy) {
  if (is_terminal(state)) {
    return 0.0;
  }

  double backed_up_value = 0.0;
  for (int action_index = 0; action_index < kActionCount; ++action_index) {
    const double action_probability = policy[state][action_index];
    const Action action = static_cast<Action>(action_index);
    for (const Outcome& outcome : transition_outcomes(state, action)) {
      backed_up_value +=
          action_probability * outcome.probability *
          (outcome.reward + kDiscount * values[outcome.next_state]);
    }
  }
  return backed_up_value;
}

std::vector<IterationRecord> evaluate_policy(const Policy& policy) {
  ValueTable values{};
  std::vector<IterationRecord> history;

  for (int iteration = 1; iteration <= kMaximumIterations; ++iteration) {
    ValueTable next_values = values;  // synchronous/Jacobi-style update
    double maximum_update = 0.0;

    for (int state = 0; state < kStateCount; ++state) {
      next_values[state] =
          bellman_expectation_backup(state, values, policy);
      maximum_update = std::max(
          maximum_update, std::abs(next_values[state] - values[state]));
    }

    values = next_values;
    history.push_back({iteration, values, maximum_update});
    if (maximum_update < kConvergenceTolerance) {
      return history;
    }
  }

  throw std::runtime_error("Policy evaluation did not converge");
}

double bellman_residual(const ValueTable& values, const Policy& policy) {
  double maximum_residual = 0.0;
  for (int state = 0; state < kStateCount; ++state) {
    maximum_residual = std::max(
        maximum_residual,
        std::abs(values[state] -
                 bellman_expectation_backup(state, values, policy)));
  }
  return maximum_residual;
}

void write_convergence_csv(const std::vector<IterationRecord>& history,
                           const std::filesystem::path& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  output << "iteration,v_terminal_left,v_state_1,v_state_2,v_state_3,"
            "v_terminal_right,maximum_update\n";
  output << std::setprecision(14);
  for (const IterationRecord& record : history) {
    output << record.iteration;
    for (double value : record.values) {
      output << ',' << value;
    }
    output << ',' << record.maximum_update << '\n';
  }
}

void write_report(const ValueTable& values,
                  int iterations,
                  double residual,
                  const std::filesystem::path& output_path) {
  std::ofstream output(output_path);
  if (!output) {
    throw std::runtime_error("Could not open " + output_path.string());
  }

  output << "Week 4 finite-MDP policy-evaluation report\n\n"
         << "States: 0 and 4 terminal; 1, 2, and 3 nonterminal.\n"
         << "Actions: LEFT and RIGHT.\n"
         << "Dynamics: 0.8 intended movement, 0.2 remain in place.\n"
         << "Reward: -1 per nonterminal transition.\n"
         << "Policy: pi(LEFT|s)=pi(RIGHT|s)=0.5.\n"
         << "Discount gamma: " << kDiscount << "\n\n"
         << "Converged values:\n" << std::fixed << std::setprecision(10);
  for (int state = 0; state < kStateCount; ++state) {
    output << "  V_pi(" << state << ") = " << values[state] << '\n';
  }
  output << "\nIterations: " << iterations << '\n'
         << std::scientific << std::setprecision(3)
         << "Final Bellman residual: " << residual << '\n';
}

}  // namespace

int main() {
  constexpr char kCsvPath[] = "build/finite_mdp_convergence.csv";
  constexpr char kReportPath[] = "build/finite_mdp_report.txt";

  try {
    const Policy policy = make_fixed_policy();
    const std::vector<IterationRecord> history = evaluate_policy(policy);
    const ValueTable& values = history.back().values;
    const double residual = bellman_residual(values, policy);

    // Symmetry is an independent structural check for this line world.
    if (std::abs(values[1] - values[3]) > kConvergenceTolerance ||
        std::abs(values[0]) > kConvergenceTolerance ||
        std::abs(values[4]) > kConvergenceTolerance ||
        residual > 2.0 * kConvergenceTolerance) {
      throw std::runtime_error("The evaluated value function failed validation");
    }

    std::filesystem::create_directories("build");
    write_convergence_csv(history, kCsvPath);
    write_report(values,
                 history.back().iteration,
                 residual,
                 kReportPath);

    std::cout << "Finite line-world MDP\n"
              << "Fixed policy: pi(left|s)=pi(right|s)=0.5\n"
              << "Discount: " << kDiscount << '\n'
              << "Iterations: " << history.back().iteration << '\n'
              << std::fixed << std::setprecision(8)
              << "V_pi = [";
    for (int state = 0; state < kStateCount; ++state) {
      if (state > 0) {
        std::cout << ", ";
      }
      std::cout << values[state];
    }
    std::cout << "]\n"
              << std::scientific << std::setprecision(3)
              << "Bellman residual: " << residual << '\n'
              << "CSV: " << kCsvPath << '\n'
              << "Report: " << kReportPath << '\n';
  } catch (const std::exception& exception) {
    std::cerr << "Error: " << exception.what() << '\n';
    return 1;
  }

  return 0;
}

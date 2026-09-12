/* Week 5: Sutton–Barto Example 4.1, with explicit Bellman backups.
   Read transition -> action_value -> evaluate_policy -> improve_policy ->
   policy_iteration -> value_iteration. No sampled episodes are needed. */
#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr int kSide = 4;
constexpr int kStates = kSide * kSide;
constexpr int kActions = 4;
constexpr double kGamma = 1.0;
constexpr double kTolerance = 1e-12;
constexpr double kTieTolerance = 1e-9;
constexpr int kMaxSweeps = 100000;
using Values = std::array<double, kStates>;
using Policy = std::array<std::array<double, kActions>, kStates>;
enum Action { Up, Right, Down, Left };
struct Outcome { int next; double reward; };
struct Solution { Values values; Policy policy; int rounds; int sweeps; };

void require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
bool terminal(int s) { return s == 0 || s == kStates - 1; }

// Environment model
Outcome transition(int s, int a) {
  if (terminal(s)) return {s, 0.0};
  constexpr int dr[] = {-1, 0, 1, 0};
  constexpr int dc[] = {0, 1, 0, -1};
  const int r = s / kSide + dr[a], c = s % kSide + dc[a];
  const bool outside = r < 0 || r >= kSide || c < 0 || c >= kSide;
  return {outside ? s : r * kSide + c, -1.0};
}

Policy uniform_policy() {
  Policy policy{};
  for (int s = 0; s < kStates; ++s)
    if (!terminal(s)) policy[s].fill(1.0 / kActions);
  return policy;
}

// At gamma=1 an arbitrary policy can loop forever with return -infinity.
// In this finite chain, every state having a positive-probability path to a
// terminal excludes a closed nonterminal recurrent class (and implies properness).
void validate_policy(const Policy& policy) {
  std::array<bool, kStates> reaches_terminal{};

  for (int s = 0; s < kStates; ++s) {
    double sum = 0;
    for (double p : policy[s]) {
      require(std::isfinite(p) && p >= 0, "Invalid action probability");
      sum += p;
    }
    require(std::abs(sum - (terminal(s) ? 0.0 : 1.0)) < 1e-12,
            "Policy row is not normalized");
    reaches_terminal[s] = terminal(s);
  }

  if (kGamma < 1.0) return;
  for (int pass = 0; pass < kStates; ++pass)
    for (int s = 0; s < kStates; ++s)
      for (int a = 0; a < kActions; ++a)
        if (policy[s][a] > 0 && reaches_terminal[transition(s, a).next])
          reaches_terminal[s] = true;

  for (bool reachable : reaches_terminal)
    require(reachable, "Undiscounted evaluation requires a terminating policy");
}

// The transition is deterministic: the sum over outcomes has one term.
double action_value(int s, int a, const Values& values) {
  const auto outcome = transition(s, a);
  return outcome.reward + kGamma * values[outcome.next];
}
// Bellman expectation equation
double expectation_backup(int s, const Values& values, const Policy& policy) {
  if (terminal(s)) return 0.0;
  double result = 0;
  for (int a = 0; a < kActions; ++a)
    result += policy[s][a] * action_value(s, a, values);
  return result;
}
double optimality_backup(int s, const Values& values) {
  if (terminal(s)) return 0.0;
  double best = action_value(s, Up, values);
  for (int a = 1; a < kActions; ++a)
    best = std::max(best, action_value(s, a, values));
  return best;
}
double residual(const Values& values, const Policy* policy) {
  double result = 0;
  for (int s = 0; s < kStates; ++s) {
    const double backed_up = policy ? expectation_backup(s, values, *policy)
                                    : optimality_backup(s, values);
    require(std::isfinite(values[s]) && std::isfinite(backed_up),
            "Nonfinite Bellman value");
    result = std::max(result, std::abs(backed_up - values[s]));
  }
  return result;
}
void log_sweep(std::ostream* log, const char* algorithm, int round, int sweep,
               double delta, const Values& values, const Policy* policy) {
  if (!log) return;
  *log << algorithm << ',' << round << ',' << sweep << ',' << delta
       << ',' << residual(values, policy);
  for (double v : values) *log << ',' << v;
  *log << '\n';
}

int evaluate_policy(const Policy& policy, Values& values,
                    std::ostream* log = nullptr, const char* label = "evaluation",
                    int round = 0) {
  validate_policy(policy);

  for (int sweep = 1; sweep <= kMaxSweeps; ++sweep) {
    Values next{};  // two arrays: every backup reads the previous sweep
    double delta = 0;
    for (int s = 0; s < kStates; ++s) {
      next[s] = expectation_backup(s, values, policy);
      delta = std::max(delta, std::abs(next[s] - values[s]));
    }
    values = next;
    log_sweep(log, label, round, sweep, delta, values, &policy);
    if (delta < kTolerance && residual(values, &policy) < kTolerance) return sweep;
  }
  throw std::runtime_error("Policy evaluation exceeded sweep limit");
}

// Share probability equally among ALL maximizing actions. A fixed tie
// tolerance gives reproducible policies despite evaluation roundoff.
Policy improve_policy(const Values& values) {
  Policy greedy{};
  for (int s = 0; s < kStates; ++s) {
    if (terminal(s)) continue;
    const double best = optimality_backup(s, values);
    int count = 0;
    for (int a = 0; a < kActions; ++a)
      if (best - action_value(s, a, values) <= kTieTolerance) ++count;
    for (int a = 0; a < kActions; ++a)
      if (best - action_value(s, a, values) <= kTieTolerance)
        greedy[s][a] = 1.0 / count;
  }
  return greedy;
}
Solution policy_iteration(std::ostream& log) {
  Policy policy = uniform_policy();  // proper, unlike e.g. always-UP
  Values values{};
  int total_sweeps = 0;
  for (int round = 1; round <= 100; ++round) {
    const Values previous = values;
    total_sweeps += evaluate_policy(policy, values, &log, "policy_iteration", round);
    if (round > 1)
      for (int s = 0; s < kStates; ++s)
        require(values[s] >= previous[s] - 1e-8, "Policy improvement decreased value");
    const Policy improved = improve_policy(values);
    if (improved == policy) return {values, policy, round, total_sweeps};
    policy = improved;  // warm-start the next evaluation with current values
  }
  throw std::runtime_error("Policy iteration exceeded round limit");
}
Solution value_iteration(std::ostream& log) {
  Values values{};
  for (int sweep = 1; sweep <= kMaxSweeps; ++sweep) {
    Values next{};
    double delta = 0;
    for (int s = 0; s < kStates; ++s) {
      next[s] = optimality_backup(s, values);
      delta = std::max(delta, std::abs(next[s] - values[s]));
    }
    values = next;
    log_sweep(&log, "value_iteration", 0, sweep, delta, values, nullptr);
    if (delta < kTolerance && residual(values, nullptr) < kTolerance)
      return {values, improve_policy(values), 0, sweep};
  }
  throw std::runtime_error("Value iteration exceeded sweep limit");
}
std::string actions(const Policy& policy, int s) {
  if (terminal(s)) return "T";
  std::string result;
  for (int a = 0; a < kActions; ++a)
    if (policy[s][a] > 0) result += "URDL"[a];
  return result;
}
void print_grid(std::ostream& out, const Values& values, const Policy& policy) {
  out << std::fixed << std::setprecision(4);
  for (int r = 0; r < kSide; ++r) {
    for (int c = 0; c < kSide; ++c) out << std::setw(10) << values[r*kSide+c];
    out << "    ";
    for (int c = 0; c < kSide; ++c) out << std::setw(6) << actions(policy, r*kSide+c);
    out << '\n';
  }
}
}  // namespace

int main() {
  try {
    std::filesystem::create_directories("build");
    std::ofstream log("build/gridworld_convergence.csv");
    std::ofstream report("build/gridworld_report.txt");
    std::ofstream table("build/gridworld_values.csv");
    require(log && report && table, "Cannot create Gridworld output files");
    log << std::setprecision(17) << "algorithm,round,sweep,maximum_update,bellman_residual";
    for (int s = 0; s < kStates; ++s) log << ",v" << s;
    log << '\n';

    const Policy random = uniform_policy();
    Values evaluated{};
    const int sweeps = evaluate_policy(random, evaluated, &log);
    const Solution pi = policy_iteration(log), vi = value_iteration(log);

    // Independent oracles: the textbook random-policy table and shortest paths.
    constexpr Values expected = {0,-14,-20,-22, -14,-18,-20,-20,
                                -20,-20,-18,-14, -22,-20,-14,0};
    // Test the environment model
    require(transition(1, Up).next == 1 && transition(1, Up).reward == -1,
            "Boundary moves must still cost -1");
    require(transition(1, Left).next == 0 && transition(1, Left).reward == -1,
            "Entering termination must cost -1");
    // Test an invalid undiscounted policy
    if (kGamma == 1.0) {
      Policy improper{};
      for (int s = 0; s < kStates; ++s) if (!terminal(s)) improper[s][Up] = 1;
      bool rejected = false;
      try { validate_policy(improper); } catch (const std::runtime_error&) { rejected = true; }
      require(rejected, "Improper undiscounted policy was accepted");
    }
    // Check optimal values with shortest paths
    for (int s = 0; s < kStates; ++s) {
      const int distance = std::min(s/kSide + s%kSide,
                                   2*(kSide-1) - s/kSide - s%kSide);
      double optimal = 0;
      for (int step = 0; step < distance; ++step) optimal -= std::pow(kGamma, step);
      if (kGamma == 1.0)
        require(std::abs(evaluated[s] - expected[s]) < 1e-8, "Textbook table mismatch");
      require(std::abs(pi.values[s] - optimal) < 1e-8 &&
              std::abs(vi.values[s] - optimal) < 1e-8, "Shortest-path value mismatch");
    }
    // Verify extracted optimal policies
    for (const Solution* solution : {&pi, &vi}) {
      Values checked{};
      evaluate_policy(solution->policy, checked);
      require(residual(solution->values, nullptr) < 1e-8, "Optimality residual too large");
      for (int s = 0; s < kStates; ++s)
        require(std::abs(checked[s] - solution->values[s]) < 1e-8,
                "Extracted policy does not achieve reported values");
    }
    table << std::setprecision(17) << "state,row,col,v_random,v_pi,v_vi,pi_actions,vi_actions\n";
    for (int s = 0; s < kStates; ++s)
      table << s << ',' << s/kSide << ',' << s%kSide << ',' << evaluated[s]
            << ',' << pi.values[s] << ',' << vi.values[s] << ','
            << actions(pi.policy, s) << ',' << actions(vi.policy, s) << '\n';
    report << "Week 5: 4x4 Gridworld, gamma=" << kGamma
           << ", reward=-1 until termination\n"
           << "T = terminal; U/R/D/L = equally probable listed actions.\n\n"
           << "Uniform policy evaluation (" << sweeps << " sweeps):\n";
    print_grid(report, evaluated, random);
    report << "\nPolicy iteration (" << pi.rounds << " evaluation/improvement rounds, "
           << pi.sweeps << " evaluation sweeps):\n";
    print_grid(report, pi.values, pi.policy);
    report << "\nValue iteration (" << vi.sweeps << " sweeps):\n";
    print_grid(report, vi.values, vi.policy);
    report << std::scientific << "\nRandom-policy residual: " << residual(evaluated, &random)
           << "\nPI optimality residual: " << residual(pi.values, nullptr)
           << "\nVI optimality residual: " << residual(vi.values, nullptr)
           << "\nAll boundary, oracle, properness and extracted-policy checks passed.\n";
    log.flush(); report.flush(); table.flush();
    require(log && report && table, "Failed writing Gridworld outputs");
    std::cout << "Gridworld checks passed. Random evaluation: " << sweeps
              << " sweeps; PI: " << pi.rounds << " rounds / " << pi.sweeps
              << " sweeps; VI: " << vi.sweeps << " sweeps.\n"
              << "Read build/gridworld_report.txt and build/gridworld_{values,convergence}.csv\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << '\n';
    return 1;
  }
}

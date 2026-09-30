#ifndef MOJOCO_STUDIO_CLIFF_LEARNING_H_
#define MOJOCO_STUDIO_CLIFF_LEARNING_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>

namespace cliff {
constexpr int rows=4,columns=12,states=48,actions=4,start=36,goal=47;
enum Action { Up,Right,Down,Left };
enum class Method { Sarsa,QLearning,ExpectedSarsa };
using Row=std::array<double,actions>;
using QTable=std::array<Row,states>;
using Values=std::array<double,states>;
using Policy=std::array<Row,states>;
struct Transition { int next; double reward; bool terminal; bool fell; };
inline bool is_cliff(int s) { return s>start && s<goal; }
inline bool active(int s) { return s>=0 && s<states && s!=goal && !is_cliff(s); }
inline const char* name(Method method) {
  return method==Method::Sarsa?"sarsa":(method==Method::QLearning?"q_learning":"expected_sarsa");
}

/**
 * @brief Sample-free deterministic dynamics of the book's 4x12 Cliff Walking.
 * State s=12*row+column, rows downward, columns rightward. Actions U,R,D,L move
 * one cell, clipped to the grid. Every transition rewards -1 except stepping
 * into bottom-row columns 1..10: reward -100, next=start, terminal=FALSE.
 * Only arrival at bottom-right goal terminates; goal-entry reward is still -1.
 * Thus falling is a transition inside the SAME episode, not a terminal reset.
 * Current state must be active; terminal/cliff cells cannot be acted from.
 * Review: RL notes 6.5, Cliff Walking (book Example 6.6). gamma=1 in experiments.
 */
inline Transition step(int s,int action) {
  if (!active(s) || action<0 || action>=actions) throw std::invalid_argument("Invalid cliff transition");
  constexpr int dr[4]={-1,0,1,0},dc[4]={0,1,0,-1};
  const int r=std::clamp(s/columns+dr[action],0,rows-1);
  const int c=std::clamp(s%columns+dc[action],0,columns-1);
  const int next=r*columns+c;
  if (is_cliff(next)) return {start,-100,false,true};
  return {next,-1,next==goal,false};
}

/**
 * @brief Explicit epsilon-greedy probabilities with UNIFORM greedy tie-breaking.
 * For G=argmax_a q[a], pi(a)=epsilon/4+(1-epsilon)*[a in G]/|G|.
 * The random branch includes greedy actions; with one maximizer and epsilon=.1
 * probabilities are .925 and .025 for each other action. All-equal q gives .25
 * each, including when epsilon=0. Exact ties use exact floating equality.
 * Input is a four-action Q row with finite values; epsilon must be in [0,1].
 * Output sums to one and is used by BOTH the sampler and Expected Sarsa target,
 * avoiding policy inconsistencies. Review: RL notes 6.4 and 6.6.
 */
inline Row epsilon_policy(const Row& q,double epsilon) {
  if (!std::isfinite(epsilon) || epsilon<0 || epsilon>1) throw std::invalid_argument("Invalid epsilon");
  const double best=*std::max_element(q.begin(),q.end());
  int ties=0;
  for (double value:q) {
    if (!std::isfinite(value)) throw std::invalid_argument("Nonfinite Q");
    ties+=value==best;
  }
  Row pi{};
  for (int a=0;a<actions;++a) pi[a]=epsilon/actions+(q[a]==best?(1-epsilon)/ties:0);
  return pi;
}

/** @brief Draw A~pi(.|s) from the explicit four-action distribution.
 * pi must be nonnegative and sum to one (as produced by epsilon_policy).
 * rng is an explicit independent training/evaluation stream; sampling is the
 * policy's randomness, not environment transition noise in this grid.
 */
inline int sample(const Row& pi,std::mt19937& rng) {
  return std::discrete_distribution<int>(pi.begin(),pi.end())(rng);
}

/** @brief Expected action value sum_a pi[a]*q[a], in reward units.
 * The four probabilities and Q entries belong to the SAME successor state and
 * are read before the current update. This averages actions, not transitions;
 * Expected Sarsa is still model-free (RL notes 6.6).
 */
inline double expectation(const Row& q,const Row& pi) {
  double value=0;
  for (int a=0;a<actions;++a) value+=pi[a]*q[a];
  return value;
}

/**
 * @brief One TD(0) state-value update from (S_t,R_{t+1},S_{t+1}).
 * delta=R_{t+1}+gamma*V_old(S_{t+1})-V_old(S_t);
 * V(S_t)<-V_old(S_t)+alpha*delta. Terminal continuation is exactly zero,
 * even if the terminal table entry contains garbage. A cliff fall is NOT
 * terminal, so it bootstraps from V(start), including when S_t=start.
 * RHS uses pre-update estimates; the returned delta exposes the prediction
 * error. Assumes 0<alpha<=1, 0<=gamma<=1 and a valid transition.
 * Policy is fixed externally for prediction. Review: RL notes 6.1.
 */
inline double td_update(Values& v,int state,const Transition& tr,double alpha,double gamma=1) {
  const double delta=tr.reward+(tr.terminal?0:gamma*v[tr.next])-v[state];
  v[state]+=alpha*delta;
  return delta;
}

/**
 * @brief One of three one-step action-value TD updates, with terminal masking.
 * Let q'=Q_old(S_{t+1},.). Targets y are:
 * Sarsa: R+gamma*q'[A_{t+1}], where A_{t+1} is sampled BEFORE this update and
 * must be carried forward and executed by the caller (especially for self-loops).
 * Q-learning: R+gamma*max_a q'[a], a greedy TARGET despite exploratory behavior.
 * Expected Sarsa: R+gamma*sum_a pi_old(a|S_{t+1})q'[a], where pi_old is the same
 * epsilon-greedy rule used by behavior, hence this experiment is on-policy.
 * All terminal targets are just R. Compute the full target before mutating
 * Q(state,action), then Q<-Q+alpha*(y-Q). Return delta=y-Q_old.
 * Inputs next_action is used only by Sarsa; all arrays are 48x4, row-major by
 * state then U,R,D,L. Assumes valid indices, 0<alpha<=1 and 0<=gamma<=1.
 * Fixed alpha/epsilon are finite-run comparisons, not convergence guarantees.
 * Review: RL notes 6.4-6.6. No trajectory importance ratio is needed for the
 * Q-learning maximum; its successor action is evaluated rather than sampled.
 */
inline double control_update(QTable& q,int state,int action,const Transition& tr,Method method,
                             int next_action,double alpha,double epsilon,double gamma=1) {
  double continuation=0;
  if (!tr.terminal) {
    const auto& row=q[tr.next];
    if (method==Method::Sarsa) continuation=row[next_action];
    else if (method==Method::QLearning) continuation=*std::max_element(row.begin(),row.end());
    else continuation=expectation(row,epsilon_policy(row,epsilon));
  }
  const double delta=tr.reward+gamma*continuation-q[state][action];
  q[state][action]+=alpha*delta;
  return delta;
}

struct Episode { double reward=0; int length=0,falls=0; bool complete=false; };

/**
 * @brief Train one complete episode with immediate, nonbootstrapped-terminal TD.
 * Behavior is epsilon-greedy with uniform ties. Sarsa selects and retains the
 * next action using pre-update Q. Other methods select their next BEHAVIOR
 * action after updating Q; their backup still read the pre-update successor
 * row. A fall returns to start but preserves the ongoing return and length.
 * The 100000-step guard THROWS rather than calling a truncated episode terminal
 * or inventing a zero continuation. Returns undiscounted accumulated reward,
 * transition count and cliff falls; gamma=1, alpha=.5, epsilon=.1 by default.
 */
inline Episode train_episode(QTable& q,Method method,std::mt19937& rng,double alpha=.5,double epsilon=.1) {
  Episode result;
  int state=start,action=sample(epsilon_policy(q[state],epsilon),rng);
  for (int t=0;t<100000;++t) {
    const auto tr=step(state,action);
    int next_action=0;
    if (!tr.terminal && method==Method::Sarsa) next_action=sample(epsilon_policy(q[tr.next],epsilon),rng);
    control_update(q,state,action,tr,method,next_action,alpha,epsilon);
    result.reward+=tr.reward; ++result.length; result.falls+=tr.fell;
    if (tr.terminal) { result.complete=true; return result; }
    state=tr.next;
    action=method==Method::Sarsa?next_action:sample(epsilon_policy(q[state],epsilon),rng);
  }
  throw std::runtime_error("Training exceeded step guard; no truncated return was accepted");
}

/** @brief Freeze Q and execute its epsilon-greedy policy without learning.
 * epsilon=.1 measures exploratory behavior; epsilon=0 measures greedy behavior
 * with uniform ties. The complete return includes every -100 fall and resets
 * only at the goal. This is held-out evaluation, using a separate RNG stream.
 * Evaluation stops after 1000 steps if the frozen policy loops. In that case
 * complete=false and reward is only a partial sum: callers must report the
 * truncation separately and must NOT treat it as a completed episode return.
 * This evaluation cap changes neither environment terminal flags nor learning.
 */
inline Episode evaluate_episode(const QTable& q,double epsilon,std::mt19937& rng) {
  Episode result;
  int state=start;
  for (int t=0;t<1000;++t) {
    const auto tr=step(state,sample(epsilon_policy(q[state],epsilon),rng));
    result.reward+=tr.reward; ++result.length; result.falls+=tr.fell;
    if (tr.terminal) { result.complete=true; return result; }
    state=tr.next;
  }
  return result;
}

/**
 * @brief Fixed epsilon-soft safe-route policy for TD prediction, not control.
 * The preferred action is Up until row<=1, then Right; at column 11 prefer
 * Down to goal. epsilon=.1 supplies positive probability for every action.
 * This is a project-defined fixed policy, chosen to make V_pi interpretable
 * and to cover all 37 active states. It does not depend on V or learned Q.
 */
inline Policy prediction_policy() {
  Policy pi{};
  for (int s=0;s<states;++s) if (active(s)) {
    const int action=s%columns==columns-1?Down:(s/columns>1?Up:Right);
    Row preference{}; preference[action]=1;
    pi[s]=epsilon_policy(preference,.1);
  }
  return pi;
}

/**
 * @brief Independent synchronous Bellman reference for the fixed prediction policy.
 * V_new(s)=sum_a pi(a|s)[r(s,a)+I(nonterminal)*V_old(next)]. Each sweep reads
 * only the old vector; goal and unoccupiable cliff entries remain zero.
 * With gamma=1 there is no discounted contraction argument. Here the policy
 * has positive action probabilities on a finite grid, and goal is reachable
 * from every active state: absorption occurs almost surely with finite mean
 * time, so the transient system has a unique finite value solution.
 * Stop at maximum change <1e-11 or throw at 100000 sweeps. This is a numerical
 * model-based reference for testing TD(0); TD receives no model expectations.
 * Review: RL notes Chapter 4 backups and 6.1 prediction versus TD sampling.
 */
inline Values reference_values(const Policy& pi) {
  Values v{};
  for (int iteration=0;iteration<100000;++iteration) {
    Values next{}; double residual=0;
    for (int s=0;s<states;++s) if (active(s)) {
      for (int a=0;a<actions;++a) {
        const auto tr=step(s,a);
        next[s]+=pi[s][a]*(tr.reward+(tr.terminal?0:v[tr.next]));
      }
      residual=std::max(residual,std::abs(next[s]-v[s]));
    }
    v=next;
    if (residual<1e-11) return v;
  }
  throw std::runtime_error("Cliff Bellman reference did not converge");
}
}  // namespace cliff
#endif

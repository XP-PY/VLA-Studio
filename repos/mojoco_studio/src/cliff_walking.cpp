#include "cliff_learning.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace {
using namespace cliff;
void require(bool value,const char* message) { if (!value) throw std::runtime_error(message); }

/** @brief Equal-state prediction RMSE over the 37 occupiable nonterminal states.
 * sqrt(sum_active (V-V_ref)^2/N_active), in undiscounted reward units. Cliff
 * cells are instantaneous reset events, not visited states; goal value is zero.
 * Equal-state weighting exposes errors in rarely visited states.
 */
double rmse(const Values& v,const Values& reference) {
  double sum=0; int count=0;
  for (int s=0;s<states;++s) if (active(s)) { sum+=std::pow(v[s]-reference[s],2); ++count; }
  return std::sqrt(sum/count);
}

/**
 * @brief Meaningful regression cases for transition timing and backup policies.
 * Test nonterminal cliff resets, goal-entry reward, boundary self-loops,
 * uniform tie probabilities and three distinct successor targets. The latter
 * uses q'=[4,4,1,-1], epsilon=.2: pi=[.45,.45,.05,.05], expectation=3.6.
 * With R=2,gamma=.9, targets are 2.9 (Sarsa selects action 2), 5.6 (max),
 * and 5.24 (expectation). Terminal targets ignore deliberately nonzero Q(goal).
 * A self-loop tests that RHS estimates are read before writing the updated row.
 */
void self_checks() {
  const auto fall=step(start,Right),win=step(35,Down),wall=step(start,Left);
  require(fall.fell && !fall.terminal && fall.next==start && fall.reward==-100,"Cliff reset semantics failed");
  require(win.terminal && win.reward==-1 && win.next==goal,"Goal transition failed");
  require(!wall.terminal && wall.next==start && wall.reward==-1,"Boundary self-loop failed");
  const Row tied=epsilon_policy({4,4,1,-1},.2),uniform=epsilon_policy({0,0,0,0},.1);
  require(std::abs(tied[0]-.45)<1e-12 && tied[2]==.05 && uniform[0]==.25,"Epsilon/tie probabilities failed");
  const std::array<Method,3> methods={Method::Sarsa,Method::QLearning,Method::ExpectedSarsa};
  const std::array<double,3> expected={2.9,5.6,5.24};
  for (int i=0;i<3;++i) {
    QTable q{}; q[1]={4,4,1,-1};
    control_update(q,0,Up,{1,2,false,false},methods[i],2,1,.2,.9);
    require(std::abs(q[0][Up]-expected[i])<1e-12,"Control target mismatch");
    q[goal].fill(999);
    control_update(q,35,Down,win,methods[i],2,1,.2);
    require(q[35][Down]==-1,"Terminal action-value bootstrap was not zero");
  }
  Values v{}; v[start]=-10;
  const double delta=td_update(v,start,fall,.5);
  require(delta==-100 && v[start]==-60,"TD cliff self-loop read wrong estimates");
  v[goal]=999; td_update(v,35,win,1);
  require(v[35]==-1,"Terminal state-value bootstrap was not zero");
  QTable q{}; q[start]={4,1,0,-1};
  control_update(q,start,Up,{start,-1,false,false},Method::Sarsa,Up,.5,.1);
  require(q[start][Up]==3.5,"Sarsa self-loop target timing failed");
  int s=start,length=0;
  for (int a:std::vector<int>{Up,Right,Right,Right,Right,Right,Right,Right,Right,Right,Right,Right,Down}) {
    const auto tr=step(s,a); require(!tr.fell,"Shortest path fell"); s=tr.next; ++length;
  }
  require(s==goal && length==13,"13-step reference route failed");
  QTable looping{};
  for (auto& row:looping) row[Left]=1;
  std::mt19937 rng(1);
  const auto truncated=evaluate_episode(looping,0,rng);
  require(!truncated.complete && truncated.length==1000 && truncated.reward==-1000,
          "Evaluation truncation was mistaken for goal termination");
}

/**
 * @brief TD(0) prediction under a fixed epsilon-soft policy, gamma=1.
 * After every transition use alpha_s=(10+N(s))^-0.6, with N incremented FIRST.
 * This per-state schedule has divergent sum alpha and finite sum alpha^2 for
 * infinitely visited states, unlike the control comparison's constant alpha.
 * Those are asymptotic conditions, not a finite-error guarantee. Each complete
 * episode starts at the book's start state; falls continue within the episode.
 * Compare with independent Bellman values at checkpoints, never use them in
 * the sampled TD targets. Review: RL notes 6.1-6.2.
 */
void prediction() {
  const Policy pi=prediction_policy();
  const Values reference=reference_values(pi);
  Values v{}; std::array<long,states> visits{};
  std::mt19937 rng(8100);
  std::ofstream curve("build/cliff_prediction_learning.csv");
  curve << "episodes,rmse\n" << std::setprecision(12);
  for (int episode=1;episode<=200000;++episode) {
    int s=start; bool complete=false;
    for (int t=0;t<100000;++t) {
      const auto tr=step(s,sample(pi[s],rng));
      const double alpha=std::pow(10.0+(++visits[s]),-.6);
      td_update(v,s,tr,alpha);
      if (tr.terminal) { complete=true; break; }
      s=tr.next;
    }
    require(complete,"Prediction episode exceeded guard");
    if (episode==100 || episode==1000 || episode==10000 || episode==50000 || episode==100000 || episode==200000)
      curve << episode << ',' << rmse(v,reference) << '\n';
  }
  long minimum=std::numeric_limits<long>::max();
  std::ofstream out("build/cliff_prediction.csv");
  out << "state,row,column,active,td_value,reference_value,visits\n" << std::setprecision(12);
  for (int s=0;s<states;++s) {
    if (active(s)) minimum=std::min(minimum,visits[s]);
    out << s << ',' << s/columns << ',' << s%columns << ',' << active(s) << ',' << v[s]
        << ',' << reference[s] << ',' << visits[s] << '\n';
  }
  require(minimum>0 && rmse(v,reference)<3,"TD prediction failed coverage/reference check");
  std::cout << "TD(0): RMSE=" << rmse(v,reference) << ", minimum active-state visits=" << minimum << '\n';
}

struct Moments {
  int n=0; double mean=0,m2=0;
  /** @brief Welford update for mean and centered sum of squares across seeds.
   * n<-n+1, delta=x-old_mean, mean<-old_mean+delta/n,
   * M2<-M2+delta*(x-new_mean). Stable alternative to sum(x^2)-n*mean^2.
   * Here each sample is one independent training seed's metric, not a time step.
   */
  void add(double x) { ++n; const double delta=x-mean; mean+=delta/n; m2+=delta*(x-mean); }
  /** @brief Standard error sqrt((M2/(n-1))/n) of the across-seed mean.
   * Returns zero for n<2. Approximate 95% intervals use mean +/-1.96*SE; with
   * only 30 seeds and heavy-tailed cliff losses these are descriptive intervals.
   */
  double se() const { return n>1?std::sqrt(m2/(n-1)/n):0; }
};

/** @brief Export one seed's deterministic greedy route for a readable grid plot.
 * Ties choose the first U,R,D,L index, unlike the uniform-tie evaluation policy.
 * This is an illustrative path, not an evaluation statistic. Repeated states
 * identify a greedy loop and reject this illustration without hiding evaluation
 * truncations. The caller chooses the first seed with a successful route and
 * records that seed; the evaluation summary includes ALL training seeds.
 */
bool export_path(const QTable& q,Method method,int seed,std::ofstream& out) {
  int s=start; std::array<bool,states> seen{};
  std::ostringstream pending;
  for (int t=0;t<states;++t) {
    pending << name(method) << ',' << seed << ',' << t << ',' << s/columns << ',' << s%columns << '\n';
    if (s==goal) { out << pending.str(); return true; }
    if (seen[s]) return false;
    seen[s]=true;
    const int action=int(std::max_element(q[s].begin(),q[s].end())-q[s].begin());
    s=step(s,action).next;
  }
  return false;
}
}  // namespace

int main() {
  try {
    std::filesystem::create_directories("build");
    self_checks(); prediction();
    constexpr int runs=30,episodes=1000,evaluations=200;
    std::ofstream raw("build/cliff_training.csv"),curve("build/cliff_learning.csv"),
      evaluation("build/cliff_evaluation.csv"),summary("build/cliff_summary.csv"),
      paths("build/cliff_paths.csv"),policies("build/cliff_policies.csv");
    raw << "method,seed,episode,return,steps,falls\n";
    curve << "method,episode,mean_return,return_se,mean_falls\n";
    evaluation << "method,seed,policy,completed,truncated,mean_completed_return,mean_completed_steps,mean_completed_falls\n";
    summary << "method,metric,mean,standard_error,seeds\n";
    paths << "method,seed,step,row,column\n";
    policies << "method,seed,state,row,column,active,greedy_action,q_up,q_right,q_down,q_left\n";
    for (auto* stream:{&curve,&evaluation,&summary,&policies}) *stream << std::setprecision(12);
    for (Method method:{Method::Sarsa,Method::QLearning,Method::ExpectedSarsa}) {
      std::array<Moments,episodes> rewards{},falls{};
      Moments online,greedy_reward,behavior_reward,behavior_falls;
      Moments greedy_completion,behavior_completion;
      bool illustrated=false;
      for (int run=0;run<runs;++run) {
        QTable q{};
        const int seed=8200+1000*static_cast<int>(method)+run;
        std::mt19937 rng(seed);
        double tail=0;
        for (int episode=0;episode<episodes;++episode) {
          const auto result=train_episode(q,method,rng);
          rewards[episode].add(result.reward); falls[episode].add(result.falls);
          if (episode>=episodes-100) tail+=result.reward/100;
          raw << name(method) << ',' << seed << ',' << episode+1 << ',' << result.reward
              << ',' << result.length << ',' << result.falls << '\n';
        }
        online.add(tail);
        for (int exploratory=0;exploratory<2;++exploratory) {
          std::mt19937 eval_rng(200000+seed*2+exploratory);
          double reward=0,length=0,fall_count=0;
          int completed=0;
          for (int e=0;e<evaluations;++e) {
            const auto result=evaluate_episode(q,exploratory?.1:0,eval_rng);
            if (result.complete) {
              ++completed; reward+=result.reward; length+=result.length; fall_count+=result.falls;
            }
          }
          (exploratory?behavior_completion:greedy_completion).add(double(completed)/evaluations);
          if (completed) {
            reward/=completed; length/=completed; fall_count/=completed;
            if (exploratory) { behavior_reward.add(reward); behavior_falls.add(fall_count); }
            else greedy_reward.add(reward);
          } else reward=length=fall_count=std::numeric_limits<double>::quiet_NaN();
          evaluation << name(method) << ',' << seed << ',' << (exploratory?"epsilon_0.1":"greedy")
                     << ',' << completed << ',' << evaluations-completed << ',' << reward << ',' << length << ',' << fall_count << '\n';
        }
        if (!illustrated && export_path(q,method,seed,paths)) {
          illustrated=true;
          for (int s=0;s<states;++s) {
            policies << name(method) << ',' << seed << ',' << s << ',' << s/columns << ',' << s%columns << ',' << active(s)
                     << ',' << (std::max_element(q[s].begin(),q[s].end())-q[s].begin());
            for (double value:q[s]) policies << ',' << value;
            policies << '\n';
          }
        }
      }
      for (int episode=0;episode<episodes;++episode)
        curve << name(method) << ',' << episode+1 << ',' << rewards[episode].mean << ','
              << rewards[episode].se() << ',' << falls[episode].mean << '\n';
      for (const auto& metric:std::vector<std::pair<std::string,Moments>>{
          {"last_100_training_return",online},{"frozen_greedy_completed_return",greedy_reward},
          {"frozen_epsilon_completed_return",behavior_reward},{"frozen_epsilon_completed_falls",behavior_falls},
          {"frozen_greedy_completion_rate",greedy_completion},{"frozen_epsilon_completion_rate",behavior_completion}})
        summary << name(method) << ',' << metric.first << ',' << metric.second.mean << ','
                << metric.second.se() << ',' << metric.second.n << '\n';
      require(illustrated && greedy_completion.mean>.5,"Too few successful greedy policies; inspect training");
      std::cout << name(method) << ": last100 training=" << online.mean << ", frozen greedy="
                << greedy_reward.mean << ", frozen exploratory=" << behavior_reward.mean
                << ", exploratory falls/completed game=" << behavior_falls.mean
                << ", greedy completion=" << greedy_completion.mean << '\n';
    }
    std::cout << "All Cliff Walking checks passed; 30 seeds x 1000 episodes per method.\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

#include "blackjack.h"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

using namespace blackjack;
namespace {
void require(bool condition,const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

/** @brief Equal-state RMSE=sqrt(sum_s (V_MC(s)-v_pi(s))^2/200).
 * This weights rare soft states as heavily as frequent hard states; it is not
 * visitation-weighted error. Values and errors are dimensionless expected
 * game returns. A finite sample has noise, so this need not fall monotonically.
 */
double rmse(const Estimates& estimates,const Values& reference) {
  double error=0;
  for (int s=0;s<states;++s) error+=std::pow(estimates[s].value-reference[s],2);
  return std::sqrt(error/states);
}

/** @brief Uniform-state performance gap mean_s[v_*(s)-v_pi(s)].
 * The oracle evaluates the frozen extracted greedy policy exactly. This
 * separates policy quality from noisy Q values; it is NOT the natural-deal
 * reward distribution used for held-out evaluation, nor a training update.
 */
double policy_gap(const Oracle& oracle,const Policy& pi,const Values& optimal) {
  const auto value=oracle.evaluate(&pi);
  double gap=0;
  for (int s=0;s<states;++s) gap+=optimal[s]-value[s];
  return gap/states;
}

/**
 * @brief Frozen-policy reward and Monte Carlo standard error on fresh games.
 * With n independent deals, mean=sum G/n, sample variance
 * s^2=(sum G^2-n*mean^2)/(n-1), SE=sqrt(s^2/n). Approximate 95% intervals are
 * mean +/-1.96 SE; this measures evaluation noise, not variability between
 * training seeds. Initial natural games count even though they have no
 * decision-state visits. n>1 is assumed; no training takes place here.
 */
void evaluate_games(const std::string& name,const Policy& pi,int seed,std::ofstream& out) {
  std::mt19937 rng(seed);
  constexpr int n=200000;
  double sum=0,squares=0;
  for (int i=0;i<n;++i) {
    const double g=generate(rng,pi).outcome;
    sum+=g; squares+=g*g;
  }
  const double mean=sum/n,se=std::sqrt((squares-n*mean*mean)/(n-1)/n);
  out << name << ',' << n << ',' << mean << ',' << se << '\n';
  std::cout << name << " held-out return=" << mean << " +/- " << 1.96*se << " (approx. 95%)\n";
}

/**
 * @brief Regression examples that distinguish timing and weighting mistakes.
 * The notes' A->B->A rewards [1,2,3] yield G=[6,5,3], first V(A)=6,
 * every V(A)=4.5. IS samples (G,rho)=(3,4),(100,0) give ordinary 6 and
 * weighted 3. These synthetic rewards deliberately exceed Blackjack's bounds.
 * Ace transitions, encode/decode, discount timing, first state-action pairs,
 * target mismatch and zero-weight handling are also checked directly.
 */
void self_checks() {
  for (int s=0;s<states;++s) require(index(decode(s))==s,"State encoding failed");
  const auto ace=add_card(add_card({},1),1);
  require(ace.sum==12 && ace.usable,"Two aces failed");
  const auto hard=add_card(ace,10);
  require(hard.sum==12 && !hard.usable,"Usable-ace conversion failed");
  Episode example{{{0,Hit,1,.5},{1,Stick,2,.5},{0,Hit,3,.5}},3};
  Estimates first{},every{};
  predict(example,first,false); predict(example,every,true);
  require(first[0].value==6 && every[0].value==4.5 && first[1].value==5,
          "Chronological first-visit test failed");
  require(returns(example,.5)[0]==2.75,"Discount timing failed");
  ActionEstimates q{}; Policy pi{};
  improve(example,q,pi,.1);
  require(q[0][Hit].count==1 && q[0][Hit].value==6 && std::abs(pi[0]-.95)<1e-15,
          "First-visit action-value/epsilon-soft update failed");
  Episode pairs{{{0,Stick,1,.5},{0,Hit,2,.5}},2};
  q={}; improve(pairs,q,pi,0);
  require(q[0][Stick].count==1 && q[0][Hit].count==1,"Used states instead of state-action keys");
  ImportanceEstimate is;
  is.add(3,4); is.add(100,0);
  require(is.ordinary.value==6 && is.weighted==3,"Importance normalization failed");
  ImportanceEstimate zero; zero.add(1,0);
  require(zero.weighted==0 && zero.ordinary.count==1,"Zero-weight handling failed");
  Policy hit{}; hit.fill(1);
  Episode matching{{{0,Hit,0,.5},{1,Hit,1,.5}},1};
  require(importance_ratio(matching,hit)==4,"Missing action in state-value ratio");
  matching.steps[1].action=Stick;
  require(importance_ratio(matching,hit)==0,"Target mismatch did not zero ratio");
}

/**
 * @brief Replicated ordinary/weighted IS prediction at soft 13 versus dealer 2.
 * Each of 50 independent runs samples 10000 games from the SAME starting
 * decision state under b(Hit|s)=1/2; target pi sticks at 20. For each completed
 * game, estimate v_pi(s0) using its return and full trajectory ratio. Report
 * MSE=(1/50)sum_run(estimate-v_reference)^2 at logarithmic checkpoints.
 * The oracle supplies only the comparison value. Fixed-start conditioning
 * removes any initial-state density ratio. Finite replicated MSE is noisy,
 * particularly for ordinary IS; no assertion forces weighted to win each time.
 * Review: RL notes 5.5-5.6. Not off-policy control or per-decision IS.
 */
void importance_experiment(const Values& reference,const Policy& target) {
  const int start=index({{13,true},2});
  const std::array<int,10> checkpoints={1,10,50,100,250,500,1000,2500,5000,10000};
  std::array<double,10> ordinary{},weighted{};
  Policy behavior{}; behavior.fill(.5);
  constexpr int repetitions=50;
  for (int run=0;run<repetitions;++run) {
    std::mt19937 rng(9000+run);
    ImportanceEstimate estimate;
    std::size_t checkpoint=0;
    for (int n=1;n<=checkpoints.back();++n) {
      const auto episode=generate(rng,behavior,start);
      estimate.add(returns(episode)[0],importance_ratio(episode,target));
      if (n==checkpoints[checkpoint]) {
        ordinary[checkpoint]+=std::pow(estimate.ordinary.value-reference[start],2)/repetitions;
        weighted[checkpoint]+=std::pow(estimate.weighted-reference[start],2)/repetitions;
        ++checkpoint;
      }
    }
  }
  std::ofstream out("build/blackjack_importance.csv");
  out << "episodes,ordinary_mse,weighted_mse,reference_value,repetitions\n" << std::setprecision(12);
  for (std::size_t i=0;i<checkpoints.size();++i)
    out << checkpoints[i] << ',' << ordinary[i] << ',' << weighted[i] << ','
        << reference[start] << ',' << repetitions << '\n';
}
}  // namespace

int main() {
  try {
    std::filesystem::create_directories("build");
    self_checks();
    const Oracle oracle;
    const Policy fixed=fixed_policy();
    const Values reference=oracle.evaluate(&fixed),optimal=oracle.evaluate(nullptr);
    Policy optimal_policy{};
    for (int s=0;s<states;++s) {
      const auto q=oracle.action_values(s,optimal);
      optimal_policy[s]=q[Hit]>q[Stick]?1:0;
      require(optimal[s]+1e-12>=reference[s],"Optimal reference below fixed-policy value");
      require(optimal[s]>=-1-1e-12 && optimal[s]<=1+1e-12,"Reference outside return bounds");
    }
    require(policy_gap(oracle,optimal_policy,optimal)<1e-12,"Optimal policy extraction failed");
    std::ofstream curve("build/blackjack_learning.csv");
    curve << "method,episodes,prediction_rmse,greedy_policy_gap\n" << std::setprecision(12);

    // MC Prediction experiment
    const std::array<int,7> checkpoints={10000,50000,100000,250000,500000,1000000,2000000};
    Estimates first{},every{};
    std::mt19937 prediction_rng(7002);
    std::size_t checkpoint=0;
    for (int n=1;n<=1000000;++n) {
      // Generate actions
      const auto episode=generate(prediction_rng,fixed);
      predict(episode,first,false); predict(episode,every,true);
      if (n==checkpoints[checkpoint]) {
        curve << "prediction," << n << ',' << rmse(first,reference) << ",nan\n";
        ++checkpoint;
      }
    }
    long minimum=std::numeric_limits<long>::max();
    for (int s=0;s<states;++s) {
      minimum=std::min(minimum,first[s].count);
      require(first[s].count==every[s].count && first[s].value==every[s].value,
              "Blackjack unexpectedly revisited a full decision state");
    }
    require(minimum>0 && rmse(first,reference)<.05,"MC prediction failed coverage/reference check");
    std::cout << "Prediction: 1M games, equal-state RMSE=" << rmse(first,reference)
              << "; minimum visits=" << minimum << "; first/every agree.\n";

    // MC Control
    ActionEstimates es{},soft{};
    Policy es_policy=fixed,soft_policy{};
    soft_policy.fill(.5);
    for (int method=0;method<2;++method) {
      std::mt19937 rng(7100+method);
      auto& q=method==0?es:soft;
      auto& pi=method==0?es_policy:soft_policy;
      const int episodes=method==0?1000000:2000000;
      checkpoint=0;
      for (int n=1;n<=episodes;++n) {
        // Explicit sampling order avoids compiler-dependent argument evaluation.
        const int start=method==0?std::uniform_int_distribution<int>(0,states-1)(rng):-1;
        const int action=method==0?std::uniform_int_distribution<int>(0,1)(rng):-1;
        const auto episode=generate(rng,pi,start,action);
        improve(episode,q,pi,method==0?0:.1);
        if (n==checkpoints[checkpoint]) {
          curve << (method==0?"exploring_starts":"epsilon_soft") << ',' << n
                << ",nan," << policy_gap(oracle,greedy(q),optimal) << '\n';
          ++checkpoint;
        }
      }
      long pair_min=std::numeric_limits<long>::max();
      for (const auto& row:q) for (const auto& estimate:row) pair_min=std::min(pair_min,estimate.count);
      const double gap=policy_gap(oracle,greedy(q),optimal);
      require(pair_min>0 && gap<.03,"MC control failed coverage/reference check");
      std::cout << (method==0?"ES":"Epsilon-soft") << ": minimum pair visits=" << pair_min
                << "; greedy uniform-state value gap=" << gap << '\n';
    }
    std::ofstream table("build/blackjack_values.csv");
    table << "sum,dealer,usable_ace,first_value,every_value,visits,reference_value,optimal_value,optimal_action,es_action,soft_greedy_action,es_stick_q,es_hit_q,soft_stick_q,soft_hit_q,es_stick_visits,es_hit_visits,soft_stick_visits,soft_hit_visits\n"
          << std::setprecision(12);
    const auto es_greedy=greedy(es),soft_greedy=greedy(soft);
    for (int s=0;s<states;++s) {
      const auto state=decode(s);
      table << state.player.sum << ',' << state.dealer << ',' << state.player.usable << ','
            << first[s].value << ',' << every[s].value << ',' << first[s].count << ',' << reference[s]
            << ',' << optimal[s] << ',' << optimal_policy[s] << ',' << es_greedy[s] << ',' << soft_greedy[s]
            << ',' << es[s][Stick].value << ',' << es[s][Hit].value
            << ',' << soft[s][Stick].value << ',' << soft[s][Hit].value
            << ',' << es[s][Stick].count << ',' << es[s][Hit].count
            << ',' << soft[s][Stick].count << ',' << soft[s][Hit].count << '\n';
    }
    std::ofstream evaluation("build/blackjack_evaluation.csv");
    evaluation << "policy,games,mean_return,standard_error\n" << std::setprecision(12);
    evaluate_games("fixed_stick20",fixed,8000,evaluation);
    evaluate_games("es_greedy",es_greedy,8001,evaluation);
    evaluate_games("epsilon_greedy_deployment",soft_greedy,8002,evaluation);
    evaluate_games("epsilon_soft_behavior",soft_policy,8003,evaluation);
    evaluate_games("oracle_optimal",optimal_policy,8004,evaluation);
    importance_experiment(reference,fixed);
    std::cout << "All MC checks passed. Tables and curves: build/blackjack_*.csv\n";
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}

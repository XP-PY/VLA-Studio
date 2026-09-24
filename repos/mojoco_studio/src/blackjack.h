#ifndef MOJOCO_STUDIO_BLACKJACK_H_
#define MOJOCO_STUDIO_BLACKJACK_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

namespace blackjack {
constexpr int states=200;
enum Action { Stick=0, Hit=1 };
struct Hand { int sum=0; bool usable=false; };
struct State { Hand player; int dealer; };
using Values=std::array<double,states>;
using Policy=std::array<double,states>;  // probability of Hit, not action index
struct Step { int state; int action; double reward; double behavior_probability; };
struct Episode { std::vector<Step> steps; double outcome=0; };

/**
 * @brief Encode (sum, dealer upcard, usable ace) as one of 200 decision states.
 * index=100*usable+10*(sum-12)+(dealer-1); dealer ace=1, face cards=10.
 * The state is sufficient because draws are independent with replacement;
 * it omits the hidden dealer card. Initial natural resolution occurs BEFORE
 * a decision state, so these values condition on an active game.
 * Review: RL notes 5.1, Blackjack example. Throws on nondecision states.
 */
inline int index(const State& s) {
  if (s.player.sum<12 || s.player.sum>21 || s.dealer<1 || s.dealer>10)
    throw std::runtime_error("Invalid Blackjack decision state");
  return 100*int(s.player.usable)+10*(s.player.sum-12)+s.dealer-1;
}

/** @brief Invert index(): recover player sum, usable ace and dealer upcard.
 * Integer quotient/remainder separate the 2x10x10 table dimensions. The
 * supplied index must be in [0,200); this is a decision state, not a full deal.
 */
inline State decode(int s) {
  if (s<0 || s>=states) throw std::runtime_error("Invalid Blackjack state index");
  return {{12+(s%100)/10,s>=100},1+s%10};
}

/**
 * @brief Update a hand by adding a rank, counting an ace as 11 when possible.
 * Start with sum'=sum+(card==ace ? 11 : card). Count the old usable ace and
 * the new ace; while sum'>21 and a usable ace remains, subtract 10 to change
 * one ace's value from 11 to 1. At most one ace can remain usable in a valid
 * nonbust hand. This handles soft 12 + ten -> hard 12 and ace + ace -> soft 12.
 * card is 1..10. sum>21 denotes bust; inputs are valid unbusted aggregate hands.
 * This is the environment transition, not a learning update (RL notes 5.1).
 */
inline Hand add_card(Hand hand, int card) {
  hand.sum+=card==1?11:card;
  int aces=int(hand.usable)+int(card==1);
  while (hand.sum>21 && aces>0) { hand.sum-=10; --aces; }
  hand.usable=aces>0;
  return hand;
}

/** @brief Infinite-deck card probability: P(1..9)=1/13, P(10)=4/13.
 * Jack/queen/king share ten's value. Used ONLY by the independent DP oracle;
 * Monte Carlo algorithms receive sampled episodes and never enumerate this.
 */
inline double card_probability(int card) { return (card==10?4.0:1.0)/13.0; }

/** @brief Sample a replacement rank uniformly from 13 ranks, map 10..13 to 10.
 * This realizes card_probability(); draws do not deplete a deck. rng is passed
 * explicitly so learning, evaluation and importance experiments have separate
 * reproducible streams. Reproducibility across C++ standard libraries can vary.
 */
inline int draw(std::mt19937& rng) {
  const int rank=std::uniform_int_distribution<int>(1,13)(rng);
  return rank>10?10:rank;
}

/** @brief Fixed prediction policy pi(Hit|s)=1 for sums <20, else 0.
 * The returned 200-vector stores probabilities, including usable-ace states.
 * Review: RL notes 5.1, fixed-policy Blackjack prediction.
 */
inline Policy fixed_policy() {
  Policy pi{};
  for (int s=0;s<states;++s) pi[s]=decode(s).player.sum<20?1:0;
  return pi;
}

/**
 * @brief Sample a complete undiscounted game under frozen policy pi.
 * Each stored tuple is (S_t,A_t,R_{t+1},b(A_t|S_t)); intermediate rewards=0,
 * terminal reward=outcome in {-1,0,+1}. Terminal states have no stored action.
 * pi[s] is P(Hit|s). No Q/V update happens while this function is running.
 *
 * Natural starts deal two player cards and dealer up/hidden cards. A player's
 * initial natural 21 resolves immediately (+1, or 0 if dealer natural), with
 * NO decision-state visit. Below 12 the player is forced to draw. Otherwise
 * play follows pi; after sticking the dealer draws below 17, including soft
 * 16, and stands on soft 17. A nonnatural 21 ties a dealer 21. No extra natural
 * payout, finite-deck effects, doubling or splitting (RL notes 5.1 convention).
 *
 * start>=0 instead supplies a synthetic active decision state (ES or IS): the
 * dealer's hidden card is still an independent draw; no initial natural test
 * applies. first_action>=0 forces an ES first action sampled uniformly by the
 * caller, so its recorded probability is 1/2. Later actions follow pi.
 * At most 100 steps is an implementation guard: exceeding it throws rather
 * than treating a truncated return as complete. Legal games terminate earlier.
 */
inline Episode generate(std::mt19937& rng, const Policy& pi, int start=-1, int first_action=-1) {
  Episode episode;
  Hand player;
  int up;
  Hand dealer;
  if (start<0) {
    player=add_card(add_card({},draw(rng)),draw(rng));
    up=draw(rng); dealer=add_card(add_card({},up),draw(rng));
    if (player.sum==21) { episode.outcome=dealer.sum==21?0:1; return episode; }
    while (player.sum<12) player=add_card(player,draw(rng));
  } else {
    const State state=decode(start); player=state.player; up=state.dealer;
    dealer=add_card(add_card({},up),draw(rng));
  }
  for (int t=0;t<100;++t) {
    const int s=index({player,up});
    const bool forced=t==0 && first_action>=0;
    const int a=forced?first_action:int(std::bernoulli_distribution(pi[s])(rng));
    const double probability=forced?.5:(a==Hit?pi[s]:1-pi[s]);
    episode.steps.push_back({s,a,0,probability});
    if (a==Hit) {
      player=add_card(player,draw(rng));
      if (player.sum<=21) continue;
      episode.outcome=-1;
    } else {
      while (dealer.sum<17) dealer=add_card(dealer,draw(rng));
      episode.outcome=dealer.sum>21 || player.sum>dealer.sum?1:(player.sum==dealer.sum?0:-1);
    }
    episode.steps.back().reward=episode.outcome;
    return episode;
  }
  throw std::runtime_error("Blackjack episode did not terminate");
}

struct Mean {
  long count=0;
  double value=0;
  /** @brief Incremental arithmetic mean: N<-N+1, V<-V+(sample-V)/N.
   * Rearranging V_N=((N-1)*V_{N-1}+sample)/N avoids storing past returns.
   * Used for complete MC returns; stationary-policy convergence assumes
   * sufficient visits and finite return moments. During control, this averages
   * returns generated by changing policies, not iid samples from one policy.
   * Review: RL notes 5.1 and 5.6.
   */
  void add(double sample) { ++count; value+=(sample-value)/count; }
};
using Estimates=std::array<Mean,states>;
using ActionEstimates=std::array<std::array<Mean,2>,states>;

/**
 * @brief Backward return recursion G_t=R_{t+1}+gamma*G_{t+1}, G_T=0.
 * Returns one G_t per episode.steps[t], in chronological order. gamma=1 for
 * Blackjack; 0<=gamma<=1 is supported for synthetic review examples. This
 * reads only realized rewards, never V or Q, and hence is not bootstrapping.
 * Episode must be complete; the terminal state's continuation value is zero.
 * Review: RL notes 5.1.
 */
inline std::vector<double> returns(const Episode& episode, double gamma=1) {
  if (gamma<0 || gamma>1) throw std::runtime_error("Invalid discount");
  std::vector<double> result(episode.steps.size());
  double g=0;
  for (std::size_t t=episode.steps.size();t>0;--t) {
    g=episode.steps[t-1].reward+gamma*g; result[t-1]=g;
  }
  return result;
}

/**
 * @brief First/every-visit state-value prediction after a complete episode.
 * Compute G backward, then walk FORWARD: update V(S_t) toward G_t by the
 * sample mean if every_visit, or if no earlier S_k=S_t with k<t. A backward
 * 'seen' set would incorrectly choose last visits. first and every estimators
 * use exactly the same episode data. Blackjack has no repeated full decision
 * states, so a synthetic revisiting episode is needed to distinguish them.
 * Review: RL notes 5.1; V estimates the fixed generating policy, not v_*.
 */
inline void predict(const Episode& episode, Estimates& estimates, bool every_visit) {
  const auto g=returns(episode);
  std::array<bool,states> seen{};
  for (std::size_t t=0;t<episode.steps.size();++t) {
    const int s=episode.steps[t].state;
    if (every_visit || !seen[s]) estimates[s].add(g[t]);
    seen[s]=true;
  }
}

/**
 * @brief First-visit state-ACTION MC evaluation plus epsilon-greedy improvement.
 * After the frozen-policy episode, for each earliest pair (s,a), update
 * N(s,a) and Q(s,a)<-Q(s,a)+(G_t-Q(s,a))/N(s,a). Then choose greedy action a*,
 * breaking ties toward Stick, and set pi(Hit|s)=epsilon/2+(1-epsilon)*[a*=Hit].
 * For two actions, epsilon=.1 gives .95/.05, not .9/.1. epsilon=0 yields
 * deterministic improvement for ES, whose starting pair supplies exploration.
 * Positive fixed epsilon optimizes within epsilon-soft policies, so its
 * behavior and the extracted greedy deployment policy are distinct.
 * All returns are completed before any estimates change; each improvement
 * reads the current post-update Q row. State-action keys, not states, define
 * first visits. Review: RL notes 5.2-5.4; no model probabilities enter updates.
 */
inline void improve(const Episode& episode, ActionEstimates& q, Policy& pi, double epsilon) {
  const auto g=returns(episode);
  std::array<std::array<bool,2>,states> seen{};
  for (std::size_t t=0;t<episode.steps.size();++t) {
    const auto& step=episode.steps[t];
    if (seen[step.state][step.action]) continue;
    seen[step.state][step.action]=true;
    q[step.state][step.action].add(g[t]);
    const bool hit=q[step.state][Hit].value>q[step.state][Stick].value;
    pi[step.state]=epsilon/2+(1-epsilon)*int(hit);
  }
}

/** @brief Extract deterministic pi(Hit|s)=[Q(s,Hit)>Q(s,Stick)].
 * Equal values choose Stick. Q comes from sampled returns; this policy is used
 * for frozen held-out evaluation, independently of epsilon-soft exploration.
 */
inline Policy greedy(const ActionEstimates& q) {
  Policy pi{};
  for (int s=0;s<states;++s) pi[s]=q[s][Hit].value>q[s][Stick].value?1:0;
  return pi;
}

struct ImportanceEstimate {
  Mean ordinary;
  double cumulative_weight=0,weighted=0;
  /**
   * @brief Compare ordinary and weighted trajectory importance sampling.
   * For returns G_i and ratios rho_i, ordinary V=(sum rho_i*G_i)/n;
   * weighted V=(sum rho_i*G_i)/(sum rho_i). Incrementally, C<-C+rho and
   * V<-V+(rho/C)*(G-V). Zero-ratio episodes STILL count in ordinary n; they
   * leave weighted V unchanged. If all weights are zero, report initial 0.
   * Ordinary is unbiased under coverage/integrability but can have enormous
   * variance; the self-normalized estimator is generally biased at finite n.
   * This experiment uses one fixed initial state and gamma=1, with no discount-
   * aware or per-decision modification. Review: RL notes 5.5-5.6.
   */
  void add(double g, double rho) {
    ordinary.add(rho*g);
    cumulative_weight+=rho;
    if (rho>0) weighted+=rho/cumulative_weight*(g-weighted);
  }
};

/**
 * @brief Full state-value trajectory ratio rho=product_{t=0}^{T-1} pi(A_t|S_t)/b(A_t|S_t).
 * The CURRENT action is included for V (unlike a Q update conditional on A_t).
 * Each denominator is recorded when the action was sampled; environment
 * transition probabilities cancel because target and behavior share dynamics.
 * Requires b>0 wherever pi>0; this experiment uses uniform b=.5 and a fixed
 * deterministic target. Any target-inconsistent action makes rho=0; otherwise
 * rho=2^T. Products can overflow on long general tasks; our bounded-length
 * Blackjack trajectories are short. Review: RL notes 5.5.
 */
inline double importance_ratio(const Episode& episode, const Policy& target) {
  double rho=1;
  for (const auto& step:episode.steps) {
    if (!(step.behavior_probability>0)) throw std::runtime_error("Behavior lacks coverage");
    const double p=step.action==Hit?target[step.state]:1-target[step.state];
    rho*=p/step.behavior_probability;
  }
  return rho;
}

// A small independent exact-model oracle checks the sampled algorithms below.
// None of these probabilities or values are supplied to MC training.
class Oracle {
  using Distribution=std::array<double,6>;  // 17..21, bust
  std::array<std::array<Distribution,22>,2> cache_{};
  std::array<std::array<bool,22>,2> ready_{};
  std::array<Distribution,11> dealer_{};

  /**
   * @brief Dealer terminal distribution under the stand-on-soft-17 rule.
   * D(h)=one_hot(h) if h>=17 or bust, otherwise sum_c p(c)*D(add_card(h,c)).
   * Memoization uses (sum,usable ace). Recursion is acyclic: within an ace
   * status sums increase, or a usable ace is irreversibly lost at sums>=12.
   * Returns probabilities of 17..21 and bust; this is an enumerated expectation,
   * not MC. Model-based reference inspired by RL Chapter 4 Bellman evaluation.
   */
  Distribution dealer_distribution(Hand hand) {
    Distribution result{};
    if (hand.sum>=17) { result[hand.sum>21?5:hand.sum-17]=1; return result; }
    if (ready_[hand.usable][hand.sum]) return cache_[hand.usable][hand.sum];
    for (int c=1;c<=10;++c) {
      const auto next=dealer_distribution(add_card(hand,c));
      for (int j=0;j<6;++j) result[j]+=card_probability(c)*next[j];
    }
    ready_[hand.usable][hand.sum]=true;
    return cache_[hand.usable][hand.sum]=result;
  }
 public:
  /** @brief Marginalize the hidden card: D_up=sum_c p(c)*D([up,c]).
   * There is no dealer peek conditioning; the hidden card remains independent
   * of all player actions. Each stored distribution must sum to one.
   */
  Oracle() {
    for (int up=1;up<=10;++up)
      for (int c=1;c<=10;++c) {
        const auto dist=dealer_distribution(add_card(add_card({},up),c));
        for (int j=0;j<6;++j) dealer_[up][j]+=card_probability(c)*dist[j];
      }
    for (int up=1;up<=10;++up) {
      double sum=0; for (double p:dealer_[up]) sum+=p;
      if (std::abs(sum-1)>1e-12) throw std::runtime_error("Dealer probabilities do not sum to one");
    }
  }

  /**
   * @brief Bellman action returns for an active state, with gamma=1.
   * Q(s,Stick)=sum_z P(dealer=z|up)*reward(player,z).
   * Q(s,Hit)=sum_c p(c)*[-1 if bust else V(s')]. For nonbust hits the immediate
   * reward is zero. Natural payouts do not enter: these are already active
   * decision states, not initial two-card deals. Returns [Stick,Hit].
   * This reference enumerates transitions; MC training does not call it.
   */
  std::array<double,2> action_values(int s, const Values& v) const {
    const State state=decode(s);
    std::array<double,2> q{};
    for (int j=0;j<6;++j) {
      const int reward=j==5 || state.player.sum>17+j?1:(state.player.sum==17+j?0:-1);
      q[Stick]+=dealer_[state.dealer][j]*reward;
    }
    for (int c=1;c<=10;++c) {
      const Hand next=add_card(state.player,c);
      q[Hit]+=card_probability(c)*(next.sum>21?-1:v[index({next,state.dealer})]);
    }
    return q;
  }

  /**
   * @brief Synchronous Bellman evaluation or optimality sweeps for validation.
   * With pi, V_new(s)=(1-pi[s])*Q_V(s,Stick)+pi[s]*Q_V(s,Hit).
   * With nullptr, V_new(s)=max_a Q_V(s,a). Every backup reads the PREVIOUS
   * 200-vector. Although gamma=1 is not a discounted contraction, this game's
   * decision graph is acyclic and episodes terminate, so backups converge.
   * Stop at max_s |V_new-V|<1e-13 (numerical tolerance), or throw at 1000 sweeps.
   * RL Chapter 4 supplies this oracle; Chapter 5 learning remains model-free.
   */
  Values evaluate(const Policy* pi) const {
    Values v{};
    for (int iteration=0;iteration<1000;++iteration) {
      Values next{}; double residual=0;
      for (int s=0;s<states;++s) {
        const auto q=action_values(s,v);
        next[s]=pi?(1-(*pi)[s])*q[Stick]+(*pi)[s]*q[Hit]:std::max(q[Stick],q[Hit]);
        residual=std::max(residual,std::abs(next[s]-v[s]));
      }
      v=next;
      if (residual<1e-13) return v;
    }
    throw std::runtime_error("Reference Bellman iteration failed to converge");
  }
};
}  // namespace blackjack
#endif

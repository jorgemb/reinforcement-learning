#include <mdp/agents.h>

namespace rl::mdp{

    BasicGridworldAgent::BasicGridworldAgent(unsigned int seed)
            : m_random_engine(seed),
              m_distribution(0, ActionTraits<Action>::available_actions().size()-1),
              m_total_reward{0}{
        auto actions = ActionTraits<Action>::available_actions();
        m_actions.insert(m_actions.begin(), actions.begin(), actions.end());

        // Check if seed should be changed
        if(seed == 0){
            m_random_engine.seed(std::random_device{}());
        }
    }

    FourWayAction BasicGridworldAgent::start(const GridworldState &initial_state) {
        // Initialize reward
        m_total_reward = {};

        // Choose a random action
        return m_actions[m_distribution(m_random_engine)];
    }

    FourWayAction BasicGridworldAgent::step(const double &reward, const GridworldState &next_state) {
        m_total_reward += reward;

        // Choose a random action
        return m_actions[m_distribution(m_random_engine)];
    }

    void BasicGridworldAgent::end(const double &reward) {
        m_total_reward += reward;
    }
} // rl::mdp
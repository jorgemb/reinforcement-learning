#include <mdp/actions.h>

namespace rl::mdp{
    // FourWayActions


    const std::array<FourWayAction, 4> ActionTraits<FourWayAction>::s_actions{
        FourWayAction::LEFT,
        FourWayAction::UP,
        FourWayAction::RIGHT,
        FourWayAction::DOWN,
    }
}
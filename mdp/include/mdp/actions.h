#ifndef REINFORCEMENT_LEARNING_ACTIONS_H
#define REINFORCEMENT_LEARNING_ACTIONS_H

#include <boost/core/span.hpp>
#include <vector>
#include <optional>
#include <array>
#include <string_view>
#include <ostream>
#include <type_traits>

namespace rl::mdp {
    /// Represent action traits that different actions must be able to show.
    /// \tparam ActionType
    template<class ActionType>
    class ActionTraits {
    public:
        /// Returns a read-only span with the available actions for the type
        /// \return
        static constexpr boost::span<const ActionType> available_actions() noexcept = delete;

        /// Returns the total amount of actions available for the type
        /// \return
        static constexpr size_t total_actions() noexcept = delete;

        /// Returns a string representation given an action
        /// \param action
        /// \return
        static constexpr std::string_view to_str(const ActionType& action) noexcept = delete;

        /// Returns the ID of the action in the underlying type in the range [0, total_actions-1]
        /// \param action
        /// \return
        static constexpr size_t id(const ActionType& action) noexcept = delete;

        /// Returns an action type from a given id
        /// \param id
        /// \return
        static constexpr ActionType from_id(size_t id) = delete;

        /// Tries to parse a given string into an action
        /// \param action_str
        /// \return
//        static std::optional<ActionType> try_parse(const std::string& action_str) = delete;
    };

    /// VON NEUMANN NEIGHBORHOOD ACTIONS ///
    enum class FourWayAction: std::size_t{
        LEFT,
        UP,
        RIGHT,
        DOWN
    };

    /// Specialization for FourWayAction
    template<>
    class ActionTraits<FourWayAction>{
    public:
        /// Returns a const span of the available actions
        /// \return
        static constexpr boost::span<const FourWayAction> available_actions() noexcept{
            return {s_actions};
        }

        /// Returns the total amount of actions available for the type
        /// \return
        static constexpr size_t total_actions() noexcept { return s_actions.size(); }

        /// Returns a string representation of the action
        /// \param action
        /// \return
        static constexpr std::string_view to_str(const FourWayAction& action) noexcept {
            if (action == FourWayAction::LEFT) {
                return "LEFT";
            } else if (action == FourWayAction::UP) {
                return "UP";
            } else if (action == FourWayAction::RIGHT) {
                return "RIGHT";
            } else {
                return "LEFT";
            }
        }

        /// Returns the ID of the action in the underlying type in the range [0, total_actions-1]
        /// \param action
        /// \return
        static constexpr size_t id(const FourWayAction& action) noexcept{ return static_cast<size_t>(action); }

        /// Returns an action type from a given id
        /// \param id
        /// \return
        static constexpr FourWayAction from_id(size_t id) { return static_cast<FourWayAction>(id); }

    private:
        static const std::array<FourWayAction, 4> s_actions;
    };

    /// Output operator for FourWayAction
    /// \param os
    /// \param action
    /// \return
    std::ostream& operator<<(std::ostream& os, const FourWayAction& action);

    /// STEERING ACTIONS ///
    enum class TwoWayAction: std::size_t{
        LEFT,
        RIGHT
    };

    /// Specialization for TwoWay Action
    template<>
    class ActionTraits<TwoWayAction>{
    public:
        /// Returns a read-only span with the available actions for the type
        /// \return
        static constexpr boost::span<const TwoWayAction> available_actions() noexcept{
            return {s_actions};
        }

        /// Returns the total amount of actions available for the type
        /// \return
        static constexpr size_t total_actions() noexcept { return s_actions.size(); }

        /// Returns a string representation given an action
        /// \param action
        /// \return
        static constexpr std::string_view to_str(const TwoWayAction& action) noexcept{
            if(action == TwoWayAction::LEFT){
                return "LEFT";
            } else {
                return "RIGHT";
            }
        }

        /// Returns the ID of the action in the underlying type in the range [0, total_actions-1]
        /// \param action
        /// \return
        static constexpr size_t id(const TwoWayAction& action) noexcept {return static_cast<size_t>(action); }

        /// Returns an action type from a given id
        /// \param id
        /// \return
        static constexpr TwoWayAction from_id(size_t id) {return static_cast<TwoWayAction>(id); }

    private:
        static const std::array<TwoWayAction, 2> s_actions;
    };

    /// Output operator for TwoWayAction
    /// \param os
    /// \param action
    /// \return
    std::ostream& operator<<(std::ostream& os, const TwoWayAction& action);
}


#endif //REINFORCEMENT_LEARNING_ACTIONS_H

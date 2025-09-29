from enum import Enum
import numpy as np
import math

class Action(Enum):
    ATTACK = 1
    DEFEND = 2
    CHANGE_FORMATION = 3
    SUBSTITUTE = 4
    DO_NOTHING = 5

class ManagerAgentPBE:
    def __init__(self, team_key, max_subs=5, gamma=1.0):
        self.team_key = team_key
        self.max_subs = max_subs
        self.available_subs = 0
        self.gamma = gamma  # discount factor
        self.belief_mu_t = 0.5  # initial belief in winning

    def update_beliefs(self, state):
        opp_key = "B" if self.team_key == "A" else "A"
        dg = state.score[self.team_key] - state.score[opp_key]   # goal difference
        t = state.time
        T = 90

        beta_min, beta_max = 0.1, 1.0 # time sensitivity
        beta_t = beta_min + (beta_max - beta_min) * (t / T)
        
        x = beta_t * dg # compute belief
        mu = 1 / (1 + math.exp(-x))

        alpha = 0.2
        #self.belief_mu_t = (1-alpha)*self.belief_mu_t + alpha*mu
        self.belief_mu_t = max(0.0, min(1.0, mu))
        #print(self.belief_mu_t)

    def stage_utility(self, state): # u(S_t, a_t) #TODO: Improve this
        base_reward = 0
        goal_diff = state.score[self.team_key] - state.score["B" if self.team_key == "A" else "A"]
        time_weight = (state.time / 90)

        if goal_diff > 0:
            base_reward += 1
        elif goal_diff < 0:
            base_reward += 1
        else:
            return 0
        utility = base_reward * (0.5 + 0.5 * time_weight)
        return utility
        
    def pressure_multiplier(self, state):
        team_pos = state.teams[self.team_key].league_position
        opp_key = "B" if self.team_key == "A" else "A"
        opp_pos = state.teams[opp_key].league_position
        home = state.home  # home vs away pressure

        baseline_pressure = 0.0

        if team_pos in [1, 2]: # title race
            baseline_pressure += 0.1

        elif team_pos in [3, 4, 5]: # european places
            baseline_pressure += 0.1

        elif team_pos in [17, 18, 19, 20]: # relegation battle
            baseline_pressure += 0.2

        if abs(team_pos - opp_pos) <= 2: # rivalry
            baseline_pressure += 0.1

        if (self.team_key == "A" and not home) or (self.team_key == "B" and home): # pressure for away team
            baseline_pressure += 0.1

        baseline_pressure = min(1.0, baseline_pressure)

        goal_diff = state.score[self.team_key] - state.score[opp_key]
        time = state.time

        if goal_diff >= 0:
            dynamic_pressure = 0.0  # if not losing, no time pressure 
        else:
            midpoint = 60  # # sigmoid pressure if losing after ~60 mins
            steepness = 0.08

            normalized_time = (time - midpoint) * steepness
            dynamic_pressure = 1 / (1 + math.exp(-normalized_time))

        total_pressure = baseline_pressure + dynamic_pressure

        if goal_diff >= 2: # winning comfortablely
            total_pressure = max(0.0, total_pressure - 0.3)

        total_pressure = min(1.0, max(0.0, total_pressure))

        return total_pressure

    def expected_utility(self, state, action): # E[U(S_{t+1}) | S_t, a_t]

        adjusted_belief = self.belief_mu_t

        #print(f"\nBefore: {adjusted_belief}")

        # adjust belief based on action
        if action == Action.ATTACK:
            adjusted_belief = min(1.0, self.belief_mu_t + 0.01 + self.pressure_multiplier(state) * 0.05) 
        elif action == Action.DEFEND:
            adjusted_belief = min(1.0, self.belief_mu_t + 0.02)
        elif action == Action.CHANGE_FORMATION:
            adjusted_belief = self.belief_mu_t  
        elif action == Action.SUBSTITUTE:
            adjusted_belief = min(1.0, self.belief_mu_t + 0.03)
        elif action == Action.DO_NOTHING:
            adjusted_belief = self.belief_mu_t  # no change

        #print(f"After: {adjusted_belief}")

        stage_reward = self.stage_utility(state) # Expected utility = adjusted belief × reward for winning
        #print(f"Stage Reward: {stage_reward}")
        expected_payoff = adjusted_belief * stage_reward
        #print(f"Expected Payoff: {expected_payoff}")

        return expected_payoff

    def choose_action(self, state): # a*_t 
        self.update_beliefs(state)

        possible_actions = [
            Action.ATTACK,
            Action.DEFEND,
            Action.CHANGE_FORMATION,
            Action.DO_NOTHING
        ]

        team = state.teams[self.team_key] 
        bench = state.benches[self.team_key]

        sub_action_data = None
        if self.available_subs < self.max_subs: # if subs are avaiable
            sub_action_data = self.evaluate_substitution(team, bench)
            if sub_action_data is not None:
                possible_actions.append(Action.SUBSTITUTE)

        action_utilities = {} # expected utility for each action
        for action in possible_actions:
            expected_payoff = self.expected_utility(state, action)
            action_utilities[action] = expected_payoff

        best_action = max(action_utilities, key=action_utilities.get) # maximum expected payoff

        if best_action == Action.SUBSTITUTE and sub_action_data is not None:
            self.available_subs += 1
            return (best_action, sub_action_data)
        elif best_action == Action.CHANGE_FORMATION:
            if self.belief_mu_t < 0.4:
                return (best_action, "3-4-3")  # attack
            elif self.belief_mu_t > 0.7:
                return (best_action, "5-3-2")  # defend
            else:
                return (best_action, "4-3-3")  # default
        else:
            return (best_action, None)

    def evaluate_substitution(self, team, bench):
        for pid, player in team.players.items():
            score = 1.0 - (player.rating / 100)

            if player.injury or player.red_card > 0 or player.yellow_card >= 1:
                score += 0.5

                if player.position == "Forward":
                    score += (1 - min(player.shots / 5.0, 1.0)) * 0.4
                elif player.position == "Midfielder":
                    score += (1 - min(player.passes / 20.0, 1.0)) * 0.4
                elif player.position == "Defender":
                    score += min(player.fouls / 3.0, 1.0) * 0.4

                if score > 0.5:
                    for bid, sub in bench.items():
                        if sub.red_card == 0 and not sub.injury and sub.position == player.position:
                            return (pid, bid)

        return None
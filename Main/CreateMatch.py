from enum import Enum
import random
import numpy as np
import matplotlib.pyplot as plt

from PlayerClass import Player
from TeamClass import Team
from GameTheory import ManagerAgentPBE

# match definiton
class MatchState:
    def __init__(self, time, teams, benches, events, score, home):
        self.time = time
        self.teams = teams
        self.benches = benches
        self.score = score
        self.events = events
        self.home = home
    
class MatchEvent:
    def __init__(self, event_type, player_id, minute, team, extra_time=0):
        self.event_type = event_type
        self.player_id = player_id
        self.minute = minute
        self.extra_time = extra_time
        self.team = team

class Action(Enum):
    ATTACK = 1
    DEFEND = 2
    CHANGE_FORMATION = 3
    SUBSTITUTE = 4
    DO_NOTHING = 5

class ManagerAgent:
    def __init__(self, team_key, model=None, max_subs=5):
        self.team_key = team_key
        self.model = model
        self.max_subs = max_subs

    def choose_action(self, state: MatchState):
        team = state.teams[self.team_key]
        bench = state.benches[self.team_key]

        if team.available_subs >= 5:
            return Action.DO_NOTHING, None
        
        for pid, player in team.players.items():
            score = 1.0 - (player.rating / 100)
            if player.red_card > 0 or player.injury or player.yellow_card >= 1:

                if player.yellow_card >= 1:
                    score += 0.5
                if player.red_card > 0 or player.injury:
                    score += 1.0

                if player.position == "Forward":
                    score += (1 - min(player.shots / 5.0, 1.0)) * 0.4
                elif player.position == "Midfielder":
                    score += (1 - min(player.passes / 20.0, 1.0)) * 0.4
                elif player.position == "Defender":
                    score += min(player.fouls / 3.0, 1.0) * 0.4

                # Now look for a valid sub from the bench
                for bid, sub in bench.items():
                    if sub.red_card == 0 and not sub.injury and score > 0.5:
                        # Found a valid sub → return substitution action
                        return Action.SUBSTITUTE, (pid, bid)

        # No one needed to be subbed or no valid subs available
        return Action.DO_NOTHING, None
        

# transition to the next match state (next minute and etc.)
def transition(state: MatchState, action: Action) -> MatchState:
    new_time = state.time + 1
    new_score = state.score.copy()
    teams = state.teams

    # Update score from events at this time
    for event in state.events:
        if state.time == 60:
            goal_difference = state.score["A"] - state.score["B"]
            
            if goal_difference >= 2:
                switch_formation(state, "A", "5-3-2")  # Team A defends deeper
            elif goal_difference <= -2:
                switch_formation(state, "A", "3-4-3")
                
        if event.minute == state.time:
            if event.event_type == "goal":
                new_score[event.team] += 1
                print(f"Team {event.team} scored! (Player {event.team})")

            # Card or injury event → update player state
            player = teams[event.team].players.get(event.player_id)
            if player:
                if event.event_type == "yellow card":
                    player.yellow_card += 1
                    print(f"{player.name} (Team {event.team}) received a yellow card at {event.minute}'")
                    if player.yellow_card == 2:
                        player.red_card = 1
                        print(f"{player.name} (Team {event.team}) received two yellow cards at {event.minute}'")
                elif event.event_type == "red card":
                    player.red_card += 1
                    print(f"{player.name} (Team {event.team}) received a red card at {event.minute}'")
                elif event.event_type == "injury":
                    player.injury = True
                    print(f"{player.name} (Team {event.team}) is injured at {event.minute}'")
                elif event.event_type == "shot":
                    player.shots += 1
                    print(f"{player.name} (Team {event.team}) shot at {event.minute}'")
                elif event.event_type == "pass":
                    player.passes += 1
                    print(f"{player.name} (Team {event.team}) pass at {event.minute}'")
                elif event.event_type == "foul":
                    player.fouls += 1
                    print(f"{player.name} (Team {event.team}) foul at {event.minute}'")

    for team_key, team in state.teams.items():
        sent_off_ids = [pid for pid, p in team.players.items() if p.red_card > 0]
        for pid in sent_off_ids:
            player = team.players.get(pid)
            if player:
                print(f"{player.name} is sent off for team {team_key}. Team is down a player.")
                del team.players[pid]

    return MatchState(
        time=new_time,
        teams=teams,
        benches=state.benches,
        events=state.events,
        score=new_score,
        home=state.home
    )

def switch_formation(state: MatchState, team_key, new_formation):
    team = state.teams[team_key]

    # Only switch if not already in the desired formation
    if team.formation != new_formation:
        print(f"Team {team_key} switching from {team.formation} to {new_formation}")
        team.formation = new_formation

        # Now adjust player roles based on the new formation
        if new_formation in ["5-3-2", "5-4-1"]:
            mids = [p for p in team.players.values() if p.position == "Midfielder"]
            if mids:
                mids[0].position = "Defender"
                print(f"{mids[0].name} moved from Midfielder to Defender for defensive coverage.")

        elif new_formation in ["3-4-3", "4-3-3"]:
            defs = [p for p in team.players.values() if p.position == "Defender"]
            if defs:
                defs[0].position = "Midfielder"
                print(f"{defs[0].name} moved from Defender to Midfielder to boost midfield control.")

        elif new_formation in ["4-5-1"]:
            fwds = [p for p in team.players.values() if p.position == "Forward"]
            if fwds:
                fwds[0].position = "Midfielder"
                print(f"{fwds[0].name} moved from Forward to Midfielder for midfield domination.")


fomations = [
    433,
    4321,
    4312,
    343,
    442,
    640
]

# create some players
players_A = {
    1: Player("Bob", 1, "A", "Goalkeeper", 80),
    2: Player("Jeff", 2, "A", "Defender", 75),
    5: Player("Leo", 5, "A", "Defender", 82),
    6: Player("Marco", 6, "A", "Defender", 77),
    7: Player("Kai", 7, "A", "Defender", 79),
    8: Player("Julian", 8, "A", "Midfielder", 74),
    9: Player("Eli", 9, "A", "Midfielder", 81),
    10: Player("Sam", 10, "A", "Midfielder", 76),
    11: Player("Rick", 11, "A", "Forward", 83),
    12: Player("Noah", 12, "A", "Forward", 72),
    13: Player("Max", 13, "A", "Forward", 78),
}

players_B = {
    3: Player("Tom", 3, "B", "Goalkeeper", 78),
    4: Player("Dan", 4, "B", "Defender", 70),
    14: Player("Liam", 14, "B", "Defender", 79),
    15: Player("Oscar", 15, "B", "Defender", 76),
    16: Player("Jay", 16, "B", "Defender", 74),
    17: Player("Finn", 17, "B", "Midfielder", 80),
    18: Player("Luke", 18, "B", "Midfielder", 82),
    19: Player("Chris", 19, "B", "Midfielder", 77),
    20: Player("Jake", 20, "B", "Forward", 75),
    21: Player("Ben", 21, "B", "Forward", 73),
    22: Player("Ryan", 22, "B", "Forward", 81),
}

benchA = {
    14: Player("Zane", 14, "A", "Defender", 70),
    15: Player("Adam", 15, "A", "Midfielder", 72),
    16: Player("Joel", 16, "A", "Midfielder", 73),
    17: Player("Matt", 17, "A", "Midfielder", 74),
    18: Player("Dylan", 18, "A", "Forward", 71),
}

benchB = {
    23: Player("Theo", 23, "B", "Defender", 71),
    24: Player("Nate", 24, "B", "Midfielder", 73),
    25: Player("Zack", 25, "B", "Midfielder", 70),
    26: Player("Luke", 26, "B", "Midfielder", 74),
    27: Player("Ethan", 27, "B", "Forward", 75),
}

teamA = Team(players_A, possession=0.6, shots=20, shots_on_target=8, passes=400, fouls=34, league_position=1, formation="4-3-3")
teamB = Team(players_B, possession=0.4, shots=27, shots_on_target=9, passes=541, fouls=12, league_position=3, formation="4-4-2")

# example events
events = [
    MatchEvent("goal", 1, 15, "A"),
    MatchEvent("goal", 3, 16, "B"),
    MatchEvent("goal", 3, 17, "B"),
    MatchEvent("goal", 3, 18, "B"),
    MatchEvent("goal", 3, 19, "B"),
    MatchEvent("goal", 3, 20, "B"),
    MatchEvent("yellow card", 4, 30, "B"),
    MatchEvent("yellow card", 4, 88, "B"),
    MatchEvent("goal", 3, 60, "B"),
    MatchEvent("red card", 2, 75, "A"),
    MatchEvent("goal", 11, 10, "A"),
    MatchEvent("yellow card", 15, 22, "B"),
    MatchEvent("goal", 9, 39, "A"),
    MatchEvent("injury", 6, 41, "A"),
    MatchEvent("goal", 18, 57, "B"),
    MatchEvent("red card", 17, 62, "B"),
    MatchEvent("goal", 11, 68, "A"),
    MatchEvent("yellow card", 10, 70, "A"),
    MatchEvent("goal", 20, 82, "B"),
    MatchEvent("injury", 13, 85, "A"),
    MatchEvent("goal", 18, 89, "B"),

    MatchEvent("shot", 1, 10, "A"),
    MatchEvent("shot", 9, 23, "A"),
    MatchEvent("shot", 20, 45, "B"),
    MatchEvent("shot", 11, 58, "A"),
    MatchEvent("shot", 18, 78, "B"),

    MatchEvent("pass", 6, 12, "A"),
    MatchEvent("pass", 8, 14, "A"),
    MatchEvent("pass", 16, 17, "B"),
    MatchEvent("pass", 19, 50, "B"),
    MatchEvent("pass", 7, 61, "A"),
    MatchEvent("pass", 6, 63, "A"),
    MatchEvent("pass", 14, 68, "B"),

    MatchEvent("foul", 4, 20, "B"),
    MatchEvent("foul", 3, 33, "B"),
    MatchEvent("foul", 13, 55, "A"),
    MatchEvent("foul", 15, 67, "B"),
    MatchEvent("foul", 10, 70, "A"),
    MatchEvent("foul", 4, 88, "B")
]



initial_state = MatchState(
    time=0,
    teams={"A": teamA, "B": teamB},
    benches={"A": benchA, "B": benchB},
    events=events,
    score={"A": 0, "B": 0},
    home=True
)

#manager = ManagerAgent(team_key="A")
manager = ManagerAgentPBE(team_key="A")

# for plotting
action_history = []
belief_history = []
pressure_history = []
beliefs = []
minutes = []
event_minutes = []  # minutes when key events happen
event_labels = []   


# simulate match
state = initial_state
for t in range(91):
    print(f"\nMinute {state.time}")
    action = manager.choose_action(state)
    print(f"Manager chooses: {action[0].name}")

    beliefs.append(manager.belief_mu_t)
    # minutes.append(state.time)

    action_history.append(action[0].name)
    belief_history.append(manager.belief_mu_t)
    pressure_history.append(manager.pressure_multiplier(state))
    minutes.append(state.time)
    if action[0].name == "SUBSTITUTE":
        pid_out, pid_in = action[1]
        team = state.teams[manager.team_key]
        bench = state.benches[manager.team_key]
        if pid_out in team.players and pid_in in bench:
            player_out = team.players.pop(pid_out)
            player_in = bench.pop(pid_in)
            team.players[pid_in] = player_in
            team.available_subs += 1
            #manager.subs_made += 1
            state.events.append(
                            MatchEvent("substitution", pid_in, state.time, manager.team_key)
                        )
            
        print(f"Substitution by team {manager.team_key}: {player_out.name} to {player_in.name}")
    
    elif action[0].name == "CHANGE_FORMATION":
        state.events.append(
                            MatchEvent("formation_change", 100, state.time, manager.team_key)
                        )
        switch_formation(state, manager.team_key, data)
    
    # for plotting events onto a graph
    for event in state.events:
        if event.minute == state.time:
            if event.event_type == "goal":
                event_minutes.append(state.time)
                event_labels.append(f"Goal ({event.team})")
            elif event.event_type == "red card":
                event_minutes.append(state.time)
                event_labels.append(f"Red Card ({event.team})")
            elif event.event_type == "yellow card":
                event_minutes.append(state.time)
                event_labels.append(f"Yellow Card ({event.team})")
            elif event.event_type == "substitution":
                event_minutes.append(state.time)
                event_labels.append(f"Substitution ({event.team})")
            elif event.event_type == "formation_change":
                event_minutes.append(state.time)
                event_labels.append(f"Formation Change ({event.team})")

    state = transition(state, action) # transition to the next state

    team = state.teams[manager.team_key]
    bench = state.benches[manager.team_key]

    injured_players = [p for p in team.players.values() if p.injury]

    for player in injured_players:
        if team.available_subs < 5:
            # Find a valid replacement
            for pid_in, sub in bench.items():
                if not sub.injury and sub.red_card == 0:
                    pid_out = player.number
                    player_out = team.players.pop(pid_out)
                    player_in = bench.pop(pid_in)
                    team.players[pid_in] = player_in
                    team.available_subs += 1

                    state.events.append(
                        MatchEvent("substitution", pid_in, state.time, manager.team_key)
                    )
                    print(f"Injury substitution: {player_out.name} → {player_in.name}")
                    break
        else:
            print(f"{player.name} is injured and can't be replaced — team plays with one fewer.")
            del team.players[player.number]
            

    print(f"Score: {state.score}")
    #print(f"Score: {state.score}, Avg fatigue: {np.mean([p.fatigue for p in state.teams['A'].players.values()]):.2f}")
    action, data = manager.choose_action(state)


# # plotting both the belief and pressure on the same graph
# plt.figure(figsize=(16, 8))
# plt.plot(minutes, belief_history, label="Belief in Success", color="blue", linewidth=2)
# plt.plot(minutes, pressure_history, label="Pressure to Attack", color="red", linestyle="--", linewidth=2)
# plt.axhline(0.5, linestyle="--", color="grey", linewidth=1, label="Neutral Belief")

# for i, minute in enumerate(event_minutes):
#     plt.axvline(minute, color="black", linestyle=":", alpha=0.6)
#     plt.text(minute, 1.05, event_labels[i], rotation=90, verticalalignment='bottom', fontsize=8)
# plt.title("Belief vs Pressure Over the Match with Key Events", pad=60)
# plt.xlabel("Match Minute")
# plt.ylabel("Value (0-1)")
# plt.ylim(0, 1.1)
# plt.legend()
# plt.xticks(range(min(minutes), max(minutes) + 1, 5))
# plt.grid(True)
# plt.show()


# # without game events
# plt.figure(figsize=(14, 6))
# plt.plot(minutes, belief_history, label="Belief in Success", color="blue", linewidth=2)
# plt.plot(minutes, pressure_history, label="Pressure to Attack", color="red", linestyle="--", linewidth=2)
# plt.axhline(0.5, linestyle="--", color="grey", linewidth=1, label="Neutral Belief")
# plt.title("Belief vs Pressure Over the Match")
# plt.xlabel("Match Minute")
# plt.ylabel("Value (0-1)")
# plt.legend()
# plt.xticks(range(min(minutes), max(minutes) + 1, 5))
# plt.grid(True)
# plt.show()



# # plot the beliefs onto the graph with respect to time
# plt.figure(figsize=(12,6))
# plt.plot(minutes, beliefs, label="Belief in Success", linewidth=2)
# plt.axhline(0.5, color='grey', linestyle='--', label="Neutral Belief")
# plt.title("Manager Belief Evolution During Match with Perfect Bayesian Equilibrium")
# plt.xlabel("Match Minute")
# plt.ylabel("Belief in Success")
# plt.legend()
# plt.xticks(range(min(minutes), max(minutes) + 1, 5))
# #plt.ylim(0, 1)
# plt.grid(True)
# plt.show()
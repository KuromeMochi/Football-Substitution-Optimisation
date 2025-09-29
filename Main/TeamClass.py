class Team:
    def __init__(self, players, possession, shots, shots_on_target, passes, fouls, league_position, formation, available_subs=0):
        self.players = players
        self.possession = possession
        self.shots = shots
        self.shots_on_target = shots_on_target
        self.passes = passes
        self.fouls = fouls
        self.league_position = league_position
        self.formation = formation
        self.available_subs = available_subs
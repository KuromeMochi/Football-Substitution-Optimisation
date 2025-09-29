class Player:
    def __init__(self, name, number, team, position, rating):
        self.name = name
        self.number  = number
        self.team = team
        self.position = position
        self.rating = rating
        self.shots = 0
        self.passes = 0
        self.fouls = 0
        self.yellow_card = 0
        self.red_card = 0
        self.injury = False
    
    def return_stats(self):
        return [
            self.number,
            self.rating,
            self.team,
            self.yellow_card,
            self.red_card
        ]


def update_player(player: Player, events): #! i think i defunct this
    new_yellow = player.yellow_card
    new_red = player.red_card
    for event in events: 
        if event.event_type == "yellow card":
            new_yellow += 1
        if event.event_type == "red card":
            new_red += 1
        if new_yellow == 2: # two yellows
            new_red += 1
        
    return Player(
        name=player.name,
        number=player.number,
        team=player.team,
        rating=player.rating,
        shots=player.shots,
        passes=player.passes,
        fouls=player.fouls,
        yellow_card=new_yellow,
        red_card=new_red,
        injury=player.injury
    )    

    # return Player(
    #     name=player.name,
    #     number=player.number,
    #     team=player.team,
    #     rating=player.rating,
    #     fatigue=player.fatigue + player.fatigue_decay,
    #     fatigue_decay=player.fatigue_decay,
    #     yellow_card=new_yellow,
    #     red_card=new_red,
    #     injury=player.injury
    # )    
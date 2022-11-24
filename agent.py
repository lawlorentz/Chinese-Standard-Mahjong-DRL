class MahjongGBAgent:

    observation_space = None
    action_space = None
    
    def __init__(self, seatWind):
        pass
    
    '''
    Wind 0..3
    Deal XX XX ...
    Player N Draw
    Player N Gang
    Player N(me) Play XX
    Player N(me) BuGang XX
    Player N(not me) Peng
    Player N(not me) Chi XX
    Player N(me) UnPeng
    Player N(me) UnChi XX
    
    Player N Hu
    Huang
    Player N Invalid
    Draw XX
    Player N(not me) Play XX
    Player N(not me) BuGang XX
    Player N(me) Peng
    Player N(me) Chi XX
    '''
    def request2obs(self, request):
        pass
    
    '''
    Hu
    Play XX
    (An)Gang XX
    BuGang XX
    Gang
    Peng
    Chi XX
    Pass
    '''
    def action2response(self, action):
        pass
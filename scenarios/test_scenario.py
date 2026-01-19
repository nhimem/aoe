from core.map import Map
from core.army import Army
from core.unit import Knight

def create_scenario(gen1_class, gen2_class):
    # Map 20x20
    game_map = Map(20, 20)
    
    # Army 1: 1 Knight
    u1 = Knight(0, 0, (5, 5))
    a1 = Army(0, [u1], gen1_class(0))
    
    # Army 2: 1 Knight
    u2 = Knight(1, 1, (15, 15))
    a2 = Army(1, [u2], gen2_class(1))
    
    return a1, a2, game_map

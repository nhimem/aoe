import sys
import os
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.map import Map
from core.army import Army
from engine import Engine
from ai.generals import MajorDAFT, ColonelKAISER
from core.definitions import UNIT_CLASS_MAP
from core.unit import Unit

def create_army(general_class, army_id, units_config, map_size):
    """
    Creates an army with specific units.
    units_config: dict { 'UnitType': count }
    """
    units = []
    w, h = map_size
    
    # Zone de déploiement (Haut-Gauche vs Bas-Droite)
    if army_id == 0:
        start_x, start_y = 5, 5
        spawn_range = 15
    else:
        start_x, start_y = w - 5, h - 5
        spawn_range = 15

    uid_counter = army_id * 10000

    for u_type, count in units_config.items():
        u_class = UNIT_CLASS_MAP.get(u_type)
        if not u_class:
            print(f"Warning: Unknown unit type {u_type}")
            continue

        for _ in range(count):
            uid_counter += 1
            # Random position in spawn area
            px = start_x + random.uniform(-spawn_range/2, spawn_range/2)
            py = start_y + random.uniform(-spawn_range/2, spawn_range/2)
            
            # Clamp to map
            px = max(1, min(w-1, px))
            py = max(1, min(h-1, py))
            
            unit = u_class(uid_counter, army_id, (px, py))
            units.append(unit)

    return Army(army_id, units, general_class(army_id))

def run_test_case(case_name, units_config, map_size=(60, 60), max_turns=100000):
    print(f"Running Test: {case_name}")
    print(f"  Configuration: {units_config}")
    print(f"  Map: {map_size}, Max Turns: {max_turns}")

    # Create Map
    game_map = Map(map_size[0], map_size[1])

    # Create Armies
    # Army 0: ColonelKAISER (The Challenger)
    # Army 1: MajorDAFT (The Dummy)
    # Note: To be fair, we use the EXACT same composition
    army_kaiser = create_army(ColonelKAISER, 0, units_config, map_size)
    army_daft = create_army(MajorDAFT, 1, units_config, map_size)

    # Initialize Engine
    engine = Engine(game_map, army_kaiser, army_daft)

    # Run Game (Headless)
    # logic_speed=1 for max speed
    try:
        engine.run_game(max_turns=max_turns, view=None, logic_speed=16)
    except Exception as e:
        print(f"  ERROR: Simulation crashed: {e}")
        return False

    # Check Result
    winner = engine.winner
    if winner == 0:
        print("  ✅ RESOLUTION: ColonelKAISER WINS!")
        return True
    elif winner == 1:
        print("  ❌ RESOLUTION: MajorDAFT WINS (KAISER FAILED)")
        return False
    else:
        print("  ⚠️ RESOLUTION: DRAW (Time limit reached or mutual destruction)")
        return False # Draw is considered failure for "Verification that Kaiser ALWAYS wins"

def main():
    print("==================================================")
    print("   🛡️  ColonelKAISER Verification Protocol  🛡️   ")
    print("==================================================")
    print("Objet: Vérifier que ColonelKAISER bat tout le temps MajorDAFT")
    print("Conditions: Différentes armées, tailles, types d'unités (max 200/armée)")
    print("--------------------------------------------------\n")

    test_cases = [
        # 1. Petite escarmouche de chevaliers
        {
            "name": "1. Small Cavalry Skirmish (50 Knights)",
            "units": {"Knight": 50},
            "map": (60, 60),
            "turns": 1000
        },
        # 2. Bataille de rang (Archers vs Archers) - Test du kiting/focus
        {
            "name": "2. Ranged Duel (50 Crossbowmen)",
            "units": {"Crossbowman": 50},
            "map": (80, 80),
            "turns": 1500
        },
        # 3. Mêlée massive (Lanciers)
        {
            "name": "3. Mass Melee (70 Pikemen)",
            "units": {"Pikeman": 70},
            "map": (80, 80),
            "turns": 2000
        },
        # 4. Armée mixte Équilibrée (Classic AoE)
        {
            "name": "4. Balanced Army (50 Knt, 50 Pik, 50 Xbow)",
            "units": {"Knight": 50, "Pikeman": 50, "Crossbowman": 50},
            "map": (100, 100),
            "turns": 2500
        },
        # 5. Siège léger (Protection d'Onager)
        {
            "name": "5. Siege Protect (10 Onager, 40 Halberd/Pikeman)",
            "units": {"Onager": 10, "Pikeman": 40},
            "map": (100, 100),
            "turns": 3000
        },
        # 6. Grande Bataille (Max Requested: 200 units)
        {
            "name": "6. Large Scale War (75 Pikemen, 75 Crossbowmen, 75 Knights)",
            "units": {"Pikeman": 75, "Crossbowman": 75, "Knight": 75},
            "map": (150, 150),
            "turns": 5000
        }
    ]

    results = []
    params_passed = 0

    for i, case in enumerate(test_cases):
        print(f"\n[Test {i+1}/{len(test_cases)}]")
        success = run_test_case(case["name"], case["units"], case["map"], case["turns"])
        results.append((case["name"], success))
        if success:
            params_passed += 1

    print("\n==================================================")
    print("   📊  FINAL REPORT  📊   ")
    print("==================================================")
    for name, success in results:
        status = "PASSED" if success else "FAILED" 
        print(f"[{'✅' if success else '❌'}] {name}: {status}")
    
    print("--------------------------------------------------")
    print(f"Total Success Rate: {params_passed}/{len(test_cases)} ({params_passed/len(test_cases)*100:.1f}%)")
    
    if params_passed == len(test_cases):
        print(">> VERIFICATION SUCCESSFUL: ColonelKAISER is dominant.")
    else:
        print(">> VERIFICATION FAILED: ColonelKAISER needs improvement.")

if __name__ == "__main__":
    main()

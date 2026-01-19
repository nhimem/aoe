# utils/unified_loader.py

import sys
from typing import Tuple
from core.map import Map
from core.army import Army
from core.unit import Unit
from ai.general import General
from core.definitions import GENERAL_CLASS_MAP, UNIT_CLASS_MAP

def load_scenario(filepath: str, general1_name: str = "MajorDAFT", general2_name: str = "MajorDAFT") -> Tuple[Map, Army, Army]:
    """
    Charges un scénario complet depuis un fichier (.scen ou text).
    Le fichier contient (optionnellement) :
    - SIZE: W H
    - GRID: (Elevation data)
    - UNITS: (Grid-like or list) -> PROBABLY LIST is better for this custom format
      Type, X, Y, OwnerID
    - STRUCTURES: 
      Type, X, Y, OwnerID
    """
    print(f"Chargement du scénario unifié depuis {filepath}...")

    width, height = 120, 120
    elevation_data = []
    units_data = [] # (TYPE, X, Y, OWNER_ID)
    structures_data = [] # (TYPE, X, Y, OWNER_ID)
    
    current_section = None

    try:
        with open(filepath, 'r') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if line.startswith("SIZE:"):
                    parts = line.split(":", 1)[1].strip().split()
                    width, height = int(parts[0]), int(parts[1])
                    current_section = None
                
                elif line.startswith("GRID:"):
                    current_section = "GRID"
                    elevation_data = []
                
                elif line.startswith("UNITS:"):
                    current_section = "UNITS"
                
                elif line.startswith("STRUCTURES:"):
                    current_section = "STRUCTURES"

                elif current_section == "GRID":
                    row = [int(e) for e in line.split()]
                    if len(row) != width:
                         # Warning or Error? Let's implement robust read
                         # If row is shorter, fill with 0? No, let's error for now to be safe.
                         raise ValueError(f"Ligne {line_number}: Grid width mismatch.")
                    elevation_data.append(row)

                elif current_section == "UNITS" or current_section == "STRUCTURES":
                    # Format: Type, X, Y, Owner
                    parts = line.split(',')
                    if len(parts) >= 4:
                        u_type = parts[0].strip()
                        u_x = float(parts[1].strip())
                        u_y = float(parts[2].strip())
                        u_owner = int(parts[3].strip())
                        
                        target_list = units_data if current_section == "UNITS" else structures_data
                        target_list.append((u_type, u_x, u_y, u_owner))

    except FileNotFoundError:
        print(f"Erreur: Fichier scénario introuvable '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Erreur parsage '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validation Grid
    if elevation_data and len(elevation_data) != height:
         # Fill with 0 if missing rows (or error)
         print(f"Attention: Données d'élévation incomplètes. Remplissage avec 0.")
         while len(elevation_data) < height:
             elevation_data.append([0]*width)

    # 1. Create Map
    game_map = Map(width, height)
    if elevation_data:
        for y in range(height):
            for x in range(width):
                game_map.get_tile(x, y).elevation = elevation_data[y][x]
    
    # 2. Setup Armies
    # We need generals. Passed as arguments or defaults.
    gen1_cls = GENERAL_CLASS_MAP.get(general1_name)
    gen2_cls = GENERAL_CLASS_MAP.get(general2_name)
    
    if not gen1_cls or not gen2_cls:
        raise ValueError("Généraux inconnus.")

    army1 = Army(0, [], gen1_cls(0))
    army2 = Army(1, [], gen2_cls(1))
    
    all_entities = units_data + structures_data
    
    id_counter_0 = 0
    id_counter_1 = 10000
    
    for u_type, u_x, u_y, u_owner in all_entities:
        u_cls = UNIT_CLASS_MAP.get(u_type)
        if not u_cls:
            print(f"Warning: Unit type '{u_type}' not found.")
            continue
            
        # Create Unit
        if u_owner == 0:
            uid = id_counter_0
            id_counter_0 += 1
            unit = u_cls(uid, 0, (u_x, u_y))
            army1.units.append(unit)
        elif u_owner == 1:
            uid = id_counter_1
            id_counter_1 += 1
            unit = u_cls(uid, 1, (u_x, u_y))
            army2.units.append(unit)
    
    return game_map, army1, army2

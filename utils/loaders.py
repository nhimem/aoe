# utils/loaders.py
import sys
from typing import Optional

from core.map import Map
from core.army import Army
from core.unit import Unit
from ai.general import General
from core.definitions import GENERAL_CLASS_MAP, UNIT_CLASS_MAP

def load_map_from_file(filepath: str) -> Map:
    """
    Charge une carte (taille, élévation, obstacles) depuis un fichier .map.
    """
    print(f"Chargement de la carte depuis {filepath}...")

    width, height = 120, 120  # Valeur par défaut
    reading_grid = False
    grid_data = []

    try:
        with open(filepath, 'r') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue

                if line.startswith("SIZE:"):
                    parts = line.split(":", 1)[1].strip().split()
                    if len(parts) != 2:
                        raise ValueError(f"Ligne {line_number}: Format SIZE invalide.")
                    width, height = int(parts[0]), int(parts[1])

                elif line.startswith("GRID:"):
                    reading_grid = True
                    grid_data = []

                elif reading_grid:
                    # Lecture des données de la grille d'élévation
                    row = [int(e) for e in line.split()]
                    if len(row) != width:
                        raise ValueError(f"Ligne {line_number}: La largeur de la grille ne correspond pas à la taille attendue de {width}.")
                    grid_data.append(row)

    except FileNotFoundError:
        print(f"Erreur: Fichier carte introuvable '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Erreur dans le fichier carte '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)

    if len(grid_data) != height:
        raise ValueError(f"La hauteur de la grille ({len(grid_data)}) ne correspond pas à la taille attendue de {height}.")

    # Créer et peupler la carte
    game_map = Map(width, height)
    for y in range(height):
        for x in range(width):
            elevation = grid_data[y][x]
            tile = game_map.get_tile(x, y)
            if tile:
                tile.elevation = elevation

    print(f"Carte chargée: {width}x{height}.")
    return game_map

def load_army_from_file(filepath: str, army_id: int, general_name: Optional[str] = None) -> Army:
    """
    Charge une définition d'armée depuis un fichier .txt
    """
    print(f"Chargement de l'armée {army_id} depuis {filepath}...")

    general_class: Optional[type[General]] = None
    units_to_create: list[tuple[type[Unit], tuple[float, float]]] = []

    try:
        with open(filepath, 'r') as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()

                if not line or line.startswith('#'):
                    continue

                if line.startswith("GENERAL:"):
                    if general_name is None:
                        general_name = line.split(":", 1)[1].strip()
                        general_class = GENERAL_CLASS_MAP.get(general_name)
                    else:
                        general_class = GENERAL_CLASS_MAP.get(general_name)

                    if general_class is None:
                        raise ValueError(f"Ligne {line_number}: Général inconnu '{general_name}'.")

                else:
                    parts = line.split(',')
                    if len(parts) != 3:
                        raise ValueError(f"Ligne {line_number}: Format invalide. Attendu: Type, X, Y")

                    unit_type_name = parts[0].strip()
                    unit_class = UNIT_CLASS_MAP.get(unit_type_name)

                    if unit_class is None:
                        raise ValueError(f"Ligne {line_number}: Type d'unité inconnu '{unit_type_name}'.")

                    pos = (float(parts[1].strip()), float(parts[2].strip()))
                    units_to_create.append((unit_class, pos))

    except FileNotFoundError:
        print(f"Erreur: Fichier armée introuvable '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
        print(f"Erreur dans le fichier '{filepath}': {e}", file=sys.stderr)
        sys.exit(1)

    if general_class is None:
        print(f"Erreur: Le fichier '{filepath}' ne définit aucun 'GENERAL:'.", file=sys.stderr)
        sys.exit(1)

    general_instance = general_class(army_id=army_id)

    created_units: list[Unit] = []

    unit_id_base = army_id * 1000

    for i, (unit_class, pos) in enumerate(units_to_create):
        unit_id = unit_id_base + i
        unit = unit_class(unit_id=unit_id, army_id=army_id, pos=pos)
        created_units.append(unit)

    print(f"Armée {army_id} chargée: {general_instance.__class__.__name__} avec {len(created_units)} unités.")
    return Army(army_id=army_id, units=created_units, general=general_instance)

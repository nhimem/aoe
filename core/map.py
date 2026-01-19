# core/map.py
import math
from core.unit import Unit

class Tile:
    """
    Représente une seule tuile sur la grille de la carte.
    """
    def __init__(self, terrain_type: str = "plain", elevation: int = 0):
        if not 0 <= elevation <= 16:
            raise ValueError("L'élévation doit être comprise entre 0 et 16.")
        self.terrain_type = terrain_type
        self.elevation = elevation
        self.units: list[Unit] = []

class Map:
    """
    Gère le monde du jeu sur une grille 2D. Chaque tuile a des propriétés
    comme le terrain et l'élévation, et contient les unités présentes.
    """
    def __init__(self, width: int, height: int):
        self.width: int = width
        self.height: int = height
        self.grid: list[list[Tile]] = [[Tile() for _ in range(height)] for _ in range(width)]
        self.obstacles: list[tuple[str, int, int]] = []

    def add_obstacle(self, type_name: str, x: int, y: int):
        """Ajoute un obstacle à la carte."""
        # Pour l'instant, on se contente de le stocker.
        # Pourrait modifier la tuile, par ex. la rendre non-traversable.
        self.obstacles.append((type_name, x, y))

    def get_tile(self, x: int, y: int) -> Tile | None:
        """Retourne l'objet Tile à une coordonnée de grille donnée."""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[x][y]
        return None

    def get_elevation_at_pos(self, pos: tuple[float, float]) -> int:
        """Retourne l'élévation à une coordonnée de position donnée."""
        x, y = int(pos[0]), int(pos[1])
        tile = self.get_tile(x, y)
        return tile.elevation if tile else 0

    def add_unit(self, unit: Unit):
        """Ajoute une unité à la grille en se basant sur sa position."""
        x, y = int(unit.pos[0]), int(unit.pos[1])
        tile = self.get_tile(x, y)
        if tile and unit not in tile.units:
            tile.units.append(unit)

    def remove_unit(self, unit: Unit):
        """Retire une unité de la grille."""
        x, y = int(unit.pos[0]), int(unit.pos[1])
        tile = self.get_tile(x, y)
        if tile and unit in tile.units:
            tile.units.remove(unit)

    def update_unit_position(self, unit: Unit, old_pos: tuple[float, float], new_pos: tuple[float, float]):
        """Met à jour la position d'une unité sur la grille."""
        old_x, old_y = int(old_pos[0]), int(old_pos[1])
        new_x, new_y = int(new_pos[0]), int(new_pos[1])

        unit.pos = new_pos  # Mettre à jour la position flottante de l'unité

        if old_x == new_x and old_y == new_y:
            return  # L'unité est restée sur la même tuile

        # Retirer de l'ancienne tuile
        old_tile = self.get_tile(old_x, old_y)
        if old_tile and unit in old_tile.units:
            old_tile.units.remove(unit)

        # Ajouter à la nouvelle tuile
        new_tile = self.get_tile(new_x, new_y)
        if new_tile and unit not in new_tile.units:
            new_tile.units.append(unit)

    def get_nearby_units(self, unit: Unit, search_radius: float) -> list[Unit]:
        """
        Trouve les unités proches en scannant les tuiles adjacentes.
        """
        nearby_units = []
        radius_in_cells = int(search_radius) + 1
        unit_x, unit_y = int(unit.pos[0]), int(unit.pos[1])

        # Itérer sur un carré de tuiles autour de l'unité
        for x in range(max(0, unit_x - radius_in_cells), min(self.width, unit_x + radius_in_cells + 1)):
            for y in range(max(0, unit_y - radius_in_cells), min(self.height, unit_y + radius_in_cells + 1)):
                tile = self.grid[x][y]
                for potential_neighbor in tile.units:
                    if potential_neighbor == unit:
                        continue

                    # Vérification finale de la distance flottante exacte
                    dist = self._calculate_distance(unit.pos, potential_neighbor.pos)
                    if dist <= search_radius:
                        nearby_units.append(potential_neighbor)

        return nearby_units

    def get_units_in_radius(self, pos: tuple[float, float], search_radius: float, include_self: bool = False) -> list[Unit]:
        """
        Trouve les unités proches autour d'une position flottante en scannant
        uniquement les tuiles pertinentes. Utilise la comparaison sur les
        distances au carré pour éviter les appels coûteux à sqrt.
        """
        nearby_units: list[Unit] = []
        radius_in_cells = int(search_radius) + 1
        unit_x, unit_y = int(pos[0]), int(pos[1])

        search_radius_sq = search_radius * search_radius

        for x in range(max(0, unit_x - radius_in_cells), min(self.width, unit_x + radius_in_cells + 1)):
            for y in range(max(0, unit_y - radius_in_cells), min(self.height, unit_y + radius_in_cells + 1)):
                tile = self.grid[x][y]
                for potential in tile.units:
                    if not include_self and (int(potential.pos[0]) == unit_x and int(potential.pos[1]) == unit_y):
                        # coarse filter; still check exact position below
                        pass

                    dx = potential.pos[0] - pos[0]
                    dy = potential.pos[1] - pos[1]
                    if dx * dx + dy * dy <= search_radius_sq:
                        nearby_units.append(potential)

        return nearby_units

    @staticmethod
    def _calculate_distance(pos1: tuple[float, float], pos2: tuple[float, float]) -> float:
        """Calcule la distance euclidienne."""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def to_dict(self) -> dict:
        """Sérialise la carte en dictionnaire."""
        grid_data = []
        for x in range(self.width):
            row = []
            for y in range(self.height):
                tile = self.grid[x][y]
                # Optimisation: ne sauvegarder que si non par défaut
                if tile.terrain_type != "plain" or tile.elevation != 0:
                     row.append({'x': x, 'y': y, 't': tile.terrain_type, 'e': tile.elevation})
            if row:
                grid_data.extend(row)
        
        return {
            'width': self.width,
            'height': self.height,
            'grid': grid_data, # Sparse representation
            'obstacles': self.obstacles
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Map':
        """Reconstruit la carte depuis un dictionnaire."""
        width = data['width']
        height = data['height']
        new_map = cls(width, height)
        
        # Restaurer la grille (sparse)
        for tile_data in data.get('grid', []):
            x, y = tile_data['x'], tile_data['y']
            if 0 <= x < width and 0 <= y < height:
                new_map.grid[x][y].terrain_type = tile_data.get('t', 'plain')
                new_map.grid[x][y].elevation = tile_data.get('e', 0)
        
        new_map.obstacles = [tuple(obs) for obs in data.get('obstacles', [])]
        return new_map

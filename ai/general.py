# ai/general.py
from core.map import Map
from core.unit import Unit
from typing import Optional
import abc # (Utilisation de classes abstraites, sec 27)

# Type alias pour les actions
Action = tuple[str, int, any]

class General(abc.ABC):
    """
    Classe de base abstraite pour une IA (req 3).
    Doit implémenter 'decide_actions'.
    """
    def __init__(self, army_id: int):
        self.army_id = army_id
    
    @abc.abstractmethod
    def decide_actions(self, current_map: Map, my_units: list[Unit], enemy_units: list[Unit]) -> list[Action]:
        """
        L'IA doit retourner une liste d'actions pour ses unités.
        """
        pass

    def find_closest_enemy(self, unit: Unit, enemy_units: list[Unit]) -> Optional[Unit]:
        """
        Une fonction utilitaire que les IA peuvent réutiliser.
        """
        closest_enemy = None
        min_dist_sq = float('inf')

        for enemy in enemy_units:
            if not enemy.is_alive:
                continue

            # Compare distances au carré pour éviter les sqrt coûteux
            try:
                dist_sq = unit._center_squared_distance(enemy)
            except Exception:
                # Fallback to slower method if helper missing
                dist = unit._calculate_distance(enemy)
                dist_sq = dist * dist

            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_enemy = enemy

        return closest_enemy
# ai/generals_impl.py
from ai.general import General, Action
from core.map import Map
from core.unit import Unit
import random
import math

# (req 3)
class CaptainBRAINDEAD(General):
    """
    IA Niveau 0: Le bon capitaine revient d'une lobotomie réussie.
    Il n'est pas en état de donner des ordres tactiques.
    Les unités agissent individuellement : elles attaquent les ennemis
    dans leur champ de vision (Line of Sight) et s'en approchent si besoin,
    mais ne cherchent pas le combat si elles sont laissées seules.
    """
    def decide_actions(self, current_map: Map, my_units: list[Unit], enemy_units: list[Unit]) -> list[Action]:
        actions = []
        
        for unit in my_units:
            # 1. "Individual action": Recherche uniquement dans la ligne de vue de l'unité.
            # Contrairement à MajorDAFT, cette IA n'utilise JAMAIS la liste globale 'enemy_units'
            # pour trouver des cibles lointaines (pas de "seek out a fight").
            nearby_enemies = current_map.get_units_in_radius(unit.pos, unit.line_of_sight)
            
            # Filtre pour ne garder que les ennemis
            nearby_enemies = [e for e in nearby_enemies if e.army_id != self.army_id]

            if nearby_enemies:
                # 2. Si des ennemis sont visibles, l'unité engage le plus proche.
                closest_enemy = self.find_closest_enemy(unit, nearby_enemies)
                
                if closest_enemy:
                    if unit.can_attack(closest_enemy):
                        # L'unité attaque si elle est à portée.
                        actions.append(("attack", unit.unit_id, closest_enemy.unit_id))
                    else:
                        # L'unité s'approche pour engager l'ennemi qu'elle voit.
                        actions.append(("move", unit.unit_id, closest_enemy.pos))
        return actions

class MajorDAFT(General):
    """
    IA simple: s'approche de l'ennemi le plus proche et attaque.
    """
    def decide_actions(self, current_map: Map, my_units: list[Unit], enemy_units: list[Unit]) -> list[Action]:
        actions = []
        if not enemy_units:
            return []

        for unit in my_units:
            # Recherche optimisée dans la ligne de vue de l'unité
            nearby_enemies = current_map.get_units_in_radius(unit.pos, unit.line_of_sight)
            
            # Filtre pour ne garder que les vrais ennemis
            nearby_enemies = [e for e in nearby_enemies if e.army_id != self.army_id]

            if not nearby_enemies:
                # Si personne n'est visible, cherche l'ennemi le plus proche sur toute la carte
                closest_enemy = self.find_closest_enemy(unit, enemy_units)
            else:
                # Sinon, se concentre sur l'ennemi visible le plus proche
                closest_enemy = self.find_closest_enemy(unit, nearby_enemies)

            if not closest_enemy:
                continue

            if closest_enemy and unit.can_attack(closest_enemy):
                actions.append(("attack", unit.unit_id, closest_enemy.unit_id))
            else:
                actions.append(("move", unit.unit_id, closest_enemy.pos))
                
        return actions

class ColonelKAISER(General):
    """
    IA Stratégique (Req 3).
    - Évaluation des menaces (counters)
    - Formations (archers derrière)
    - Kiting pour les unités à distance
    - Focus fire intelligent
    - Cohésion: Les mêlées attendent les tireurs si trop éloignées
    """
    # Constantes pour la configuration de l'IA
    KITING_RANGE_PERCENTAGE = 0.5  # Les unités à distance fuient si un ennemi est à 50% de leur portée
    MELEE_ATTACK_RANGE = 2  # Portée maximale pour être considéré comme une unité de mêlée
    RANGED_FORMATION_OFFSET = 2  # Les unités à distance essaient de rester à 2 unités derrière le centre
    TARGET_EVALUATION_RANGE_MULTIPLIER = 1.5  # Ne considère que les cibles dans 150% de la ligne de vue
    LOW_HP_BONUS_MULTIPLIER = 1.5  # Bonus de score pour attaquer les cibles qui peuvent être tuées en 2 coups
    EPSILON = 0.1  # Pour éviter la division par zéro
    MELEE_WAIT_DISTANCE = 15  # Distance à laquelle les mêlées s'arrêtent pour attendre
    MELEE_RESUME_DISTANCE = 10  # Distance à laquelle les mêlées reprennent la marche (hystérésis)

    def __init__(self, army_id: int):
        super().__init__(army_id)
        self.target_memory: dict[int, int] = {}
        self.waiting_units: set[int] = set()  # IDs des unités actuellement en attente

    def _is_unit_in_combat(self, unit: Unit, enemy_units: list[Unit]) -> bool:
        """Vérifie si une unité est en combat (ennemi à portée d'attaque)."""
        for enemy in enemy_units:
            if enemy.is_alive and unit.can_attack(enemy):
                return True
        return False

    def _calculate_group_centroid(self, units: list[Unit]) -> tuple[float, float] | None:
        """Calcule le centroïde d'un groupe d'unités."""
        if not units:
            return None
        avg_x = sum(u.pos[0] for u in units) / len(units)
        avg_y = sum(u.pos[1] for u in units) / len(units)
        return (avg_x, avg_y)

    def _get_distance_to_ranged(self, melee_unit: Unit, ranged_units: list[Unit]) -> float | None:
        """Calcule la distance entre une unité de mêlée et le centroïde des tireurs."""
        ranged_centroid = self._calculate_group_centroid(ranged_units)
        if ranged_centroid is None:
            return None
        dx = melee_unit.pos[0] - ranged_centroid[0]
        dy = melee_unit.pos[1] - ranged_centroid[1]
        return math.sqrt(dx**2 + dy**2)

    def _should_melee_wait(self, melee_unit: Unit, ranged_units: list[Unit], enemy_units: list[Unit]) -> bool:
        """
        Détermine si une unité de mêlée doit attendre les tireurs.
        Utilise une hystérésis pour éviter l'oscillation:
        - S'arrête quand distance > MELEE_WAIT_DISTANCE
        - Reprend seulement quand distance < MELEE_RESUME_DISTANCE
        """
        if not ranged_units:
            self.waiting_units.discard(melee_unit.unit_id)
            return False  # Pas de tireurs, pas besoin d'attendre
        
        if self._is_unit_in_combat(melee_unit, enemy_units):
            self.waiting_units.discard(melee_unit.unit_id)
            return False  # En combat, on n'attend pas
        
        distance = self._get_distance_to_ranged(melee_unit, ranged_units)
        if distance is None:
            return False
        
        is_currently_waiting = melee_unit.unit_id in self.waiting_units
        
        if is_currently_waiting:
            # Si déjà en attente, ne reprendre que si distance < seuil de reprise
            if distance < self.MELEE_RESUME_DISTANCE:
                self.waiting_units.discard(melee_unit.unit_id)
                return False  # Reprendre la marche
            return True  # Continuer à attendre
        else:
            # Si pas en attente, commencer à attendre si distance > seuil d'arrêt
            if distance > self.MELEE_WAIT_DISTANCE:
                self.waiting_units.add(melee_unit.unit_id)
                return True  # Commencer à attendre
            return False  # Continuer à avancer

    def decide_actions(self, current_map: Map, my_units: list[Unit], enemy_units: list[Unit]) -> list[Action]:
        actions = []
        if not enemy_units or not my_units:
            return []

        enemy_lookup = {u.unit_id: u for u in enemy_units}

        melee_units = [u for u in my_units if u.attack_range <= self.MELEE_ATTACK_RANGE]
        ranged_units = [u for u in my_units if u.attack_range > self.MELEE_ATTACK_RANGE]

        avg_pos_x = sum(u.pos[0] for u in my_units) / len(my_units)
        avg_pos_y = sum(u.pos[1] for u in my_units) / len(my_units)
        army_centroid = (avg_pos_x, avg_pos_y)

        for unit in my_units:
            if unit in ranged_units:
                # limiter la recherche de menace aux unités proches
                threat_candidates = current_map.get_units_in_radius(unit.pos, unit.line_of_sight)
                threat = self.find_closest_enemy(unit, threat_candidates)
                if threat:
                    dist = unit._calculate_distance(threat)
                    safety_distance = unit.attack_range * self.KITING_RANGE_PERCENTAGE

                    if dist < safety_distance and threat.attack_range <= self.MELEE_ATTACK_RANGE:
                        flee_pos = self._calculate_flee_position(unit, threat)
                        actions.append(("move", unit.unit_id, flee_pos))
                        continue

            target = self._find_best_target(unit, enemy_units, enemy_lookup, current_map)

            if target:
                if unit.can_attack(target):
                    actions.append(("attack", unit.unit_id, target.unit_id))
                else:
                    if unit in melee_units:
                        # Vérifier si l'unité de mêlée doit attendre les tireurs
                        if self._should_melee_wait(unit, ranged_units, enemy_units):
                            # Au lieu de rester immobile, se rapprocher des tireurs
                            ranged_centroid = self._calculate_group_centroid(ranged_units)
                            if ranged_centroid:
                                actions.append(("move", unit.unit_id, ranged_centroid))
                            continue
                        actions.append(("move", unit.unit_id, target.pos))
                    elif unit in ranged_units:
                        move_pos = self._calculate_formation_position(unit, target, army_centroid)
                        actions.append(("move", unit.unit_id, move_pos))
            else:
                 # Fallback: Avancer vers l'ennemi le plus proche si aucun visible
                 closest_any = self.find_closest_enemy(unit, enemy_units)
                 if closest_any:
                      # Même logique d'attente pour les mêlées dans le fallback
                      if unit in melee_units and self._should_melee_wait(unit, ranged_units, enemy_units):
                          ranged_centroid = self._calculate_group_centroid(ranged_units)
                          if ranged_centroid:
                              actions.append(("move", unit.unit_id, ranged_centroid))
                          continue
                      actions.append(("move", unit.unit_id, closest_any.pos))
                 else:
                      actions.append(("move", unit.unit_id, army_centroid))

        return actions

    def _find_best_target(self, unit: Unit, enemy_units: list[Unit], enemy_lookup: dict, current_map: Map) -> Unit | None:
        """
        Sélectionne la meilleure cible selon une heuristique de 'valeur'.
        Score = (Dégâts infligés / Dégâts reçus) * Bonus de proximité * Bonus Kill
        """
        best_target = None
        max_score = -1

        if unit.unit_id in self.target_memory:
            candidate = enemy_lookup.get(self.target_memory[unit.unit_id])
            if candidate and candidate.is_alive:
                return candidate
            else:
                del self.target_memory[unit.unit_id]

        # Limiter la recherche aux unités proches (spatial partition via la map)
        search_radius = unit.line_of_sight * self.TARGET_EVALUATION_RANGE_MULTIPLIER
        nearby_candidates = current_map.get_nearby_units(unit, search_radius)
        for enemy in nearby_candidates:
            if not enemy.is_alive or enemy.army_id == unit.army_id:
                continue

            dist = unit._calculate_distance(enemy)

            my_damage = unit.calculate_damage(enemy, current_map)
            enemy_damage = enemy.calculate_damage(unit, current_map)
            score = my_damage / (enemy_damage + self.EPSILON)

            proximity_bonus = 1 / (1 + dist)
            final_score = score * proximity_bonus

            if enemy.current_hp < my_damage * 2:
                final_score *= self.LOW_HP_BONUS_MULTIPLIER

            if final_score > max_score:
                max_score = final_score
                best_target = enemy

        if best_target:
            self.target_memory[unit.unit_id] = best_target.unit_id

        return best_target

    def _calculate_flee_position(self, unit: Unit, threat: Unit) -> tuple[float, float]:
        dx = unit.pos[0] - threat.pos[0]
        dy = unit.pos[1] - threat.pos[1]
        dist = math.sqrt(dx**2 + dy**2)
        if dist == 0: dist = self.EPSILON

        nx = unit.pos[0] + (dx / dist) * unit.speed
        ny = unit.pos[1] + (dy / dist) * unit.speed
        return (nx, ny)

    def _calculate_formation_position(self, unit: Unit, target: Unit, army_centroid: tuple) -> tuple[float, float]:
        dir_x = target.pos[0] - army_centroid[0]
        dir_y = target.pos[1] - army_centroid[1]
        dist = math.sqrt(dir_x**2 + dir_y**2)
        if dist == 0: dist = self.EPSILON

        pos_x = army_centroid[0] + (dir_x / dist) * (dist - self.RANGED_FORMATION_OFFSET)
        pos_y = army_centroid[1] + (dir_y / dist) * (dist - self.RANGED_FORMATION_OFFSET)

        return (pos_x, pos_y)
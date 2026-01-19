# engine.py
import math
import os
from typing import Optional, Any
from core.map import Map
from core.army import Army
from core.unit import Unit
from ai.general import General

# Type alias pour les actions que l'IA peut retourner
Action = tuple[str, int, Any]

# Chemin par défaut pour les sauvegardes rapides (F11/F12)
QUICK_SAVE_PATH = "saves/quicksave.sav"

class Engine:
    """
    Le moteur de jeu principal (req 5).
    Gère la boucle de jeu, l'état du monde, et met à jour la
    matrice creuse (la Map) avec les positions flottantes.
    """
    def __init__(self, map_instance: Map, army1: Army, army2: Army):
        self.map: Map = map_instance
        self.armies: list[Army] = [army1, army2]
        self.turn_count: int = 0
        self.time_elapsed: float = 0.0  # Temps écoulé en secondes
        self.game_over: bool = False
        self.winner: Optional[int] = None 
        
        # État de pause pour l'interface
        self.paused: bool = False 

        # Dictionnaire central pour accès O(1) aux unités
        self.units_by_id: dict[int, Unit] = {}

        # Optimisation : Set des obstacles pour recherche O(1)
        self.obstacle_set = set((x, y) for _, x, y in self.map.obstacles)

        for army in self.armies:
            for unit in army.units:
                if unit.unit_id in self.units_by_id:
                    raise ValueError(f"Erreur: ID d'unité {unit.unit_id} dupliqué.")
                self.units_by_id[unit.unit_id] = unit
                self.map.add_unit(unit)

    def run_game(self, max_turns: int = 2000, view: Optional[Any] = None, logic_speed: int = 2, quiet: bool = False):
        """
        Boucle principale du jeu.
        logic_speed : Diviseur de frame.
            - Avant: 15 (4 updates/sec)
            - Maintenant: 2 (30 updates/sec) pour fluidité
        quiet: Si True, supprime les messages console (pour mode tournoi)
        """
        if not quiet:
            print(f"Debut de la partie sur une carte de {self.map.width}x{self.map.height}!")

        frame_counter = 0
        # Vitesse de la logique : Plus ce chiffre est haut, plus le jeu est lent (moins d'updates)
        # Pour du temps réel fluide, on veut ~30 updates/sec => 60FPS / 2 = 30
        LOGIC_SPEED_DIVIDER = logic_speed
        game_speed_multiplier = 1.0
        step_once = False # Pour le mode pas-à-pas (touche S)

        # Indiquer la présence d'une vue externe (pour déléguer l'animation au rendu)
        self.view_present = bool(view)
        
        # Calcul du Delta Time (Temps écoulé entre deux updates logiques)
        # Si on tourne à 60 FPS et qu'on update tous les N frames:
        # dt_base = N / 60.0 secondes
        dt_base = LOGIC_SPEED_DIVIDER / 60.0

        while not self.game_over and self.turn_count < max_turns:
            
            # --- 1. Gestion des Inputs et Affichage ---
            if view:
                # La vue nous renvoie des commandes (pause, step, quit)
                # On passe time_elapsed au lieu de turn_count
                command = view.display(self.armies, self.time_elapsed, self.paused, game_speed_multiplier)
                
                if command == "quit":
                    break
                elif command == "toggle_pause":
                    self.paused = not self.paused
                    if not quiet:
                        print(f"Jeu {'en PAUSE' if self.paused else 'REPRIS'}")
                elif command == "step":
                    self.paused = True # Le pas-à-pas force la pause après
                    step_once = True
                elif command == "quick_save":
                    # F11 - Sauvegarde rapide (Requis par le PDF)
                    # Import local pour éviter l'import circulaire
                    from utils.serialization import save_game
                    os.makedirs("saves", exist_ok=True)
                    save_game(self, QUICK_SAVE_PATH)
                    full_path = os.path.abspath(QUICK_SAVE_PATH)
                    print(f"Sauvegarde rapide effectuée: {full_path}")
                elif command == "quick_load":
                    # F12 - Chargement rapide en pleine partie
                    if os.path.exists(QUICK_SAVE_PATH):
                        from utils.serialization import load_game
                        try:
                            loaded_engine = load_game(QUICK_SAVE_PATH)
                            # Copier l'état du moteur chargé vers self
                            self.map = loaded_engine.map
                            self.armies = loaded_engine.armies
                            self.turn_count = loaded_engine.turn_count
                            self.time_elapsed = loaded_engine.time_elapsed
                            self.game_over = loaded_engine.game_over
                            self.winner = loaded_engine.winner
                            self.paused = True  # Mettre en pause après chargement
                            # Reconstruire le cache d'unités
                            self.units_by_id = {}
                            for army in self.armies:
                                for unit in army.units:
                                    self.units_by_id[unit.unit_id] = unit
                            # Mettre à jour la map dans la vue
                            if view:
                                view.map = self.map
                            print(f"Chargement reussi ! Temps: {self.time_elapsed:.1f}s")
                        except Exception as e:
                            print(f"Erreur de chargement: {e}")
                    else:
                        print(f"Aucune sauvegarde trouvee: {QUICK_SAVE_PATH}")
                elif command == "switch_view":
                    # F9 - Basculer dynamiquement entre les vues
                    try:
                        # Nettoyer la vue actuelle
                        if hasattr(view, 'cleanup'):
                            view.cleanup()
                        
                        # Determiner vers quelle vue basculer
                        from view.terminal_view import TerminalView
                        from view.gui_view import PygameView
                        
                        if isinstance(view, TerminalView):
                            # Passer vers Pygame
                            print("\nBasculement vers la vue Pygame 2.5D...")
                            view = PygameView(self.map, self.armies)
                        else:
                            # Passer vers Terminal
                            print("\nBasculement vers la vue Terminal...")
                            view = TerminalView(self.map)
                        
                        # Mettre a jour view_present
                        self.view_present = bool(view)
                    except Exception as e:
                        print(f"Erreur lors du changement de vue: {e}")
                        # En cas d'erreur, essayer de restaurer une vue terminal
                        try:
                            from view.terminal_view import TerminalView
                            view = TerminalView(self.map)
                        except:
                            pass
                elif command == "speed_up":
                    # Augmenter la vitesse de simulation (x2, x4, x8, max x16)
                    game_speed_multiplier = min(16.0, game_speed_multiplier * 2)
                    print(f"Vitesse de simulation: x{game_speed_multiplier}")
                elif command == "speed_down":
                    # Diminuer la vitesse (min x0.25)
                    game_speed_multiplier = max(0.25, game_speed_multiplier / 2)
                    print(f"Vitesse de simulation: x{game_speed_multiplier}")

            elif not quiet and self.turn_count % 30 == 0: # Moins de spam en console
                print(f"\n--- TEMPS {self.time_elapsed:.1f}s ---")

            # --- 2. Blocage si en pause ---
            if self.paused and not step_once:
                continue

            # --- 3. Ralentissement de la logique (si pas en mode step) ---
            frame_counter += 1
            if not step_once and frame_counter % LOGIC_SPEED_DIVIDER != 0:
                continue
            
            step_once = False # Reset du step

            # --- DÉBUT DE LA LOGIQUE DU TOUR ---
            self._reap_dead_units()

            if self._check_game_over():
                break

            all_actions: list[Action] = []
            for army in self.armies:
                if not army.is_defeated():
                    my_living_units = [u for u in army.units if u.is_alive]
                    enemy_living_units = self.get_enemy_units(army.army_id)
                    if not enemy_living_units: continue

                    actions = army.general.decide_actions(self.map, my_living_units, enemy_living_units)
                    all_actions.extend(actions)
            dt_sim = dt_base * game_speed_multiplier
            self._execute_actions(all_actions, dt_sim)
            self.turn_count += 1
            self.time_elapsed += dt_sim
            # --- FIN DE LA LOGIQUE ---

        if view:
            view.display(self.armies, self.time_elapsed, self.paused)
            # Afficher l'écran de fin de partie
            try:
                view.display_game_over(self.armies, self.winner, self.time_elapsed)
            except Exception:
                pass  # Si la méthode n'existe pas (mode terminal)

        if not quiet:
            print("\n--- FIN DE LA PARTIE ---")
            if self.winner is not None:
                print(f"Le vainqueur est l'Armee {self.winner + 1}!")
            elif self.turn_count >= max_turns:
                print("Limite de ticks atteinte. Egalite.")
            else:
                print("Egalite.")

    def _determine_unit_status(self, unit: Unit):
        """Updates the status of a unit based on its state and surroundings."""
        if not getattr(unit, 'is_alive', True):
            unit.statut = 'death'
            return

        prev = getattr(unit, 'statut', None)
        new_stat = prev

        # Check if attacking
        target_id = getattr(unit, 'target_id', None)
        target = self.units_by_id.get(target_id) if target_id is not None else None
        
        is_attacking = False
        if target and getattr(target, 'is_alive', False):
            if unit.can_attack(target):
                 new_stat = 'attack'
                 is_attacking = True
            else:
                 # Has target but not in range -> check if actually moving
                 # If blocked, should be idle, otherwise walk
                 if getattr(unit, 'is_moving', False):
                     new_stat = 'walk'
                 else:
                     new_stat = 'idle'
        else:
            # No target
            if getattr(unit, 'is_moving', False):
                new_stat = 'walk'
            else:
                new_stat = 'idle'

        # If played-once animation is running, don't interrupt unless death
        if getattr(unit, 'anim_play_once_remaining', 0) > 0 and new_stat != 'death':
            return

        # Apply state change
        if new_stat != prev:
            if prev != 'death': # Death is final
                unit.statut = new_stat
                try:
                    unit.anim_index = 0
                    unit.anim_play_once_remaining = 0
                except Exception:
                    pass

    def _execute_actions(self, actions: list[Action], dt: float):
        """Exécute les actions (Temps, Mouvements, Attaques)."""

        # 0. FAIRE AVANCER LE TEMPS (COOLDOWNS)
        # dt est le temps écoulé en secondes depuis la dernière update
        TIME_STEP = dt 
        for unit in self.units_by_id.values():
            # Reset moving flag at start of frame
            unit.is_moving = False
            
            # cooldown uniquement si vivant
            if unit.is_alive:
                unit.tick_cooldown(TIME_STEP)

            # Si on n'a pas de vue externe, faire avancer les animations ici.
            if not getattr(self, 'view_present', False):
                try:
                    unit.tick_animation(int(TIME_STEP * 1000))
                except Exception:
                    pass

            # Avancer le timer de la mort si l'unité est en état 'death'
            try:
                if getattr(unit, 'statut', None) == 'death':
                    unit.death_elapsed = getattr(unit, 'death_elapsed', 0) + int(TIME_STEP * 1000)
            except Exception:
                pass

        # 1. Mouvements (Prioritaires)
        move_actions = [a for a in actions if a[0] == "move"]
        
        def get_move_priority(action):
            u = self.units_by_id.get(action[1])
            if not u: return float('inf')
            t_x, t_y = action[2]
            # Priorité aux unités les plus proches de leur destination
            return (u.pos[0] - t_x)**2 + (u.pos[1] - t_y)**2

        move_actions.sort(key=get_move_priority)

        for _, unit_id, target_pos in move_actions:
            unit = self.units_by_id.get(unit_id)
            if unit and unit.is_alive:
                try:
                    unit.target_id = None
                except Exception:
                    pass
                self._handle_movement(unit, target_pos, dt)

        # 2. Attaques (Après mouvements)
        for action_type, unit_id, data in actions:
            if action_type == "attack":
                unit = self.units_by_id.get(unit_id)
                target_id = data
                target = self.units_by_id.get(target_id)
                
                # If we are attacking, we might also be moving towards the target (move due to attack range)
                # But here we handle the Attack Action (firing/striking)
                # Movement towards target is often handled by 'move' action if IA issued it,
                # OR handled implicitly if your engine does auto-move-to-attack (not the case here, separate actions).
                
                if unit and unit.is_alive and target and target.is_alive:
                    unit.target_id = target_id
                    
                    # If out of range, IA should have sent a move command separate/combined.
                    # If we are strictly in Attack logic:
                    if unit.can_attack(target) and unit.can_act():
                        final_damage = unit.calculate_damage(target, self.map)
                        target.take_damage(final_damage)
                        unit.current_cooldown = unit.reload_time
                        # Déclencher l'animation d'attaque (jouer la ligne complète)
                        try:
                            frames = getattr(unit, 'anim_frames_per_state', {}).get('attack', 30)
                            unit.anim_play_once_remaining = frames
                            unit.anim_index = 0
                            unit.statut = 'attack' # Force attack status immediately
                        except Exception:
                            pass

                        # --- Splash Damage (Onager, Trebuchet) ---
                        if hasattr(unit, 'splash_radius') and unit.splash_radius > 0:
                            splash_damage = int(final_damage * 0.5)  # 50% des dégâts
                            nearby = self.map.get_units_in_radius(target.pos, unit.splash_radius)
                            for other in nearby:
                                if other.is_alive and other != target and other.army_id != unit.army_id:
                                    other.take_damage(splash_damage)

                        # --- Trample Damage (Elite War Elephant) ---
                        if hasattr(unit, 'trample_radius') and unit.trample_radius > 0:
                            trample_dmg = int(final_damage * getattr(unit, 'trample_damage_ratio', 0.5))
                            nearby = self.map.get_units_in_radius(unit.pos, unit.trample_radius)
                            for other in nearby:
                                if other.is_alive and other != target and other.army_id != unit.army_id:
                                    other.take_damage(trample_dmg)
        
        # 3. Actions spéciales des Moines (Heal / Conversion) - PDF Req 6
        for action_type, unit_id, data in actions:
            unit = self.units_by_id.get(unit_id)
            if not unit or not unit.is_alive:
                continue
            
            # Le Monk a des attributs spéciaux
            if not hasattr(unit, 'heal_rate'):
                continue
            
            if action_type == "heal":
                # Soigner une unité alliée
                target = self.units_by_id.get(data)
                if target and target.is_alive and target.army_id == unit.army_id:
                    if unit.can_act():
                        heal_amount = unit.heal_rate
                        target.current_hp = min(target.max_hp, target.current_hp + heal_amount)
                        unit.current_cooldown = unit.reload_time
            
            elif action_type == "convert":
                # Tenter de convertir une unité ennemie
                target = self.units_by_id.get(data)
                if target and target.is_alive and target.army_id != unit.army_id:
                    distance = math.sqrt((unit.pos[0] - target.pos[0])**2 + 
                                        (unit.pos[1] - target.pos[1])**2)
                    if distance <= getattr(unit, 'conversion_range', 9.0) and unit.can_act():
                        # Conversion réussie (simplifié : toujours réussie après cooldown)
                        old_army_id = target.army_id
                        target.army_id = unit.army_id
                        unit.current_cooldown = getattr(unit, 'conversion_time', 4.0)
                        print(f"CONVERSION: {unit} a converti {target}!")

        # 4. Final Status Update
        for uid, unit in self.units_by_id.items():
            self._determine_unit_status(unit)

    def _handle_movement(self, unit: Unit, target_pos: tuple[float, float], dt: float):
        """Calcule et applique le mouvement."""
        old_pos = unit.pos
        # Conserver last_pos pour l'orientation des sprites
        try:
            unit.last_pos = old_pos
        except Exception:
            pass
        vector_x = target_pos[0] - old_pos[0]
        vector_y = target_pos[1] - old_pos[1]
        distance = math.sqrt(vector_x**2 + vector_y**2)

        if distance < 0.01:
            unit.is_moving = False
            return

        # Vitesse ajustée par le delta time

        move_dist = min(distance, unit.speed * dt)
        
        norm_x = vector_x / distance
        norm_y = vector_y / distance
        potential_pos = (old_pos[0] + norm_x * move_dist, old_pos[1] + norm_y * move_dist)

        # --- Gestion des Obstacles (Sliding) ---
        if self._is_colliding_with_obstacle(potential_pos, unit.hitbox_radius):
            # Tentative de glissement en X
            test_pos_x = (potential_pos[0], old_pos[1])
            can_move_x = not self._is_colliding_with_obstacle(test_pos_x, unit.hitbox_radius)

            # Tentative de glissement en Y
            test_pos_y = (old_pos[0], potential_pos[1])
            can_move_y = not self._is_colliding_with_obstacle(test_pos_y, unit.hitbox_radius)

            if can_move_x and can_move_y:
                # Si on peut glisser des deux côtés, on prend celui qui conserve le plus de mouvement
                if abs(vector_x) > abs(vector_y):
                     potential_pos = test_pos_x
                else:
                     potential_pos = test_pos_y
            elif can_move_x:
                potential_pos = test_pos_x
            elif can_move_y:
                potential_pos = test_pos_y
            else:
                # Bloqué complètement
                potential_pos = old_pos

        # --- Collision Detection (Unités) ---
        final_pos = self._resolve_collisions(unit, potential_pos)

        # --- Collision avec les Bords de la Map (CORRECTION) ---
        x = max(0.1, min(final_pos[0], self.map.width - 0.1))
        y = max(0.1, min(final_pos[1], self.map.height - 0.1))
        final_pos = (x, y)

        # Check if actual movement occurred (compare final_pos vs old_pos)
        # Note: If pushed by other units, you might move even if "blocked" in potential pos
        # But for 'walk' animation we usually care about intentional movement or significant displacement.
        dx = final_pos[0] - old_pos[0]
        dy = final_pos[1] - old_pos[1]
        
        # If moved significantly, mark as moving
        if dx*dx + dy*dy > 1e-6:
             unit.is_moving = True
        else:
             unit.is_moving = False

        self.map.update_unit_position(unit, old_pos, final_pos)

    def _is_colliding_with_obstacle(self, pos: tuple[float, float], radius: float) -> bool:
        """Vérifie si une position chevauche un obstacle de la carte."""
        min_x = int(pos[0] - radius)
        max_x = int(pos[0] + radius)
        min_y = int(pos[1] - radius)
        max_y = int(pos[1] + radius)

        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                if (x, y) in self.obstacle_set:
                    return True
        return False

    def _resolve_collisions(self, moving_unit: Unit, potential_pos: tuple[float, float]) -> tuple[float, float]:
        """Gère le chevauchement avec une logique de glissement pour éviter les blocages en ligne."""
        final_x, final_y = potential_pos
        nearby_units = self.map.get_nearby_units(moving_unit, search_radius=moving_unit.hitbox_radius + 2.0)

        for other_unit in nearby_units:
            if other_unit == moving_unit or not other_unit.is_alive: 
                continue

            dx = final_x - other_unit.pos[0]
            dy = final_y - other_unit.pos[1]
            dist_sq = dx*dx + dy*dy
            sum_radii = moving_unit.hitbox_radius + other_unit.hitbox_radius
            
            # Optimization: check squared distance
            if dist_sq < (sum_radii * sum_radii):
                dist = math.sqrt(dist_sq)
                
                # Prevent division by zero
                if dist < 0.001:
                    dist = 0.001
                    dx = 0.01  # Arbitrary small push
                    dy = 0.0
                
                # Calculate overlap
                # Use a small margin (0.95) to allow slight overlap and prevent jitter
                overlap = (sum_radii * 0.95) - dist
                
                if overlap > 0:
                    # Normal vector
                    nx, ny = dx / dist, dy / dist
                    
                    # Side-step vector (tangent)
                    sx, sy = -ny, nx
                    
                    # Apply push: mostly back (normal), some side (tangent) to slide
                    push_factor = 0.4
                    slide_factor = 0.6
                    
                    final_x += (nx * push_factor + sx * slide_factor) * overlap
                    final_y += (ny * push_factor + sy * slide_factor) * overlap

        # Clamp to map bounds to prevent artifacts
        final_x = max(0, min(self.map.width, final_x))
        final_y = max(0, min(self.map.height, final_y))
        
        return (final_x, final_y)
    def _reap_dead_units(self):
        """Retire les unités mortes."""
        # Si nous avons une vue (animations côté client), laisser la
        # durée de l'animation de mort s'écouler avant suppression.
        for unit_id in list(self.units_by_id.keys()):
            unit = self.units_by_id.get(unit_id)
            if not unit or unit.is_alive:
                continue

            # Mode headless / pas de vue : suppression immédiate
            if not getattr(self, 'view_present', False):
                try:
                    self.map.remove_unit(unit)
                except Exception:
                    pass
                try:
                    del self.units_by_id[unit_id]
                except Exception:
                    pass
                # Retirer également de la liste d'unités de son armée pour
                # empêcher la vue de continuer à l'afficher.
                try:
                    for army in self.armies:
                        if unit in army.units:
                            army.units.remove(unit)
                            break
                except Exception:
                    pass
                continue

            # Avec vue : attendre la fin de l'animation 'death'
            frames = getattr(unit, 'anim_frames_per_state', {}).get('death', 30)
            ms_per_frame = getattr(unit, 'anim_speed', 150)
            death_duration_ms = max(0, int(frames * ms_per_frame))
            elapsed = getattr(unit, 'death_elapsed', 0)
            # Remove when either the engine timer exceeded the expected
            # death animation duration OR when the unit reports the
            # death animation finished (robust against timing mismatches).
            if elapsed >= death_duration_ms or getattr(unit, 'death_anim_finished', False):
                try:
                    self.map.remove_unit(unit)
                except Exception:
                    pass
                try:
                   del self.units_by_id[unit_id]
                except Exception:
                   pass
                try:
                    for army in self.armies:
                        if unit in army.units:
                            army.units.remove(unit)
                            break
                except Exception:
                    pass

    def to_dict(self) -> dict:
        """Sérialise le moteur de jeu entier."""
        return {
            'map': self.map.to_dict(),
            'army1': self.armies[0].to_dict(),
            'army2': self.armies[1].to_dict(),
            'turn_count': self.turn_count,
            'time_elapsed': self.time_elapsed,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Engine':
        """Reconstruit le moteur de jeu."""
        # Note: L'ordre de reconstruction est important
        
        # 1. Map
        game_map = Map.from_dict(data['map'])
        
        # 2. Armies
        army1 = Army.from_dict(data['army1'])
        army2 = Army.from_dict(data['army2'])
        
        # 3. Engine (re-populate map spatial grid via __init__)
        engine = cls(game_map, army1, army2)
        
        # 4. Restore state
        engine.turn_count = data['turn_count']
        engine.time_elapsed = data['time_elapsed']
        
        return engine


    def _check_game_over(self) -> bool:
        """
        Vérifie les conditions de fin de partie.
        Conditions de victoire (PDF):
        1. Destruction de la Merveille (Wonder) ennemie = Victoire immédiate
        2. Élimination de toutes les unités ennemies
        """
        # Import local pour éviter les imports circulaires
        from core.unit import Wonder
        
        # Vérification de la destruction des Wonders (Condition prioritaire selon PDF)
        for army_idx, army in enumerate(self.armies):
            enemy_idx = 1 - army_idx  # L'autre armée
            
            # Vérifier si l'armée ennemie avait une Wonder et si elle est morte
            for unit in self.armies[enemy_idx].units:
                if isinstance(unit, Wonder) and not unit.is_alive:
                    # La Wonder ennemie est détruite -> Victoire !
                    self.game_over = True
                    self.winner = army.army_id
                    print(f"La Wonder de l'Armee {enemy_idx + 1} a ete detruite !")
                    return True
        
        # Vérification standard: élimination de toutes les unités
        army1_defeated = self.armies[0].is_defeated()
        army2_defeated = self.armies[1].is_defeated()
        
        if army1_defeated and army2_defeated:
            self.game_over = True
            self.winner = None
        elif army1_defeated:
            self.game_over = True
            self.winner = self.armies[1].army_id
        elif army2_defeated:
            self.game_over = True
            self.winner = self.armies[0].army_id
        
        return self.game_over

    def get_enemy_units(self, my_army_id: int) -> list[Unit]:
        return [u for u in self.units_by_id.values() if u.army_id != my_army_id]

    def get_enemy_units_near(self, my_army_id: int, pos: tuple[float, float], radius: float) -> list[Unit]:
        """Retourne la liste des unités ennemies proches d'une position donnée.

        Utilise la grille de la `Map` pour limiter la recherche aux tuiles pertinentes
        et éviter de scanner l'ensemble de `units_by_id`.
        """
        nearby = self.map.get_units_in_radius(pos, radius)
        return [u for u in nearby if u.army_id != my_army_id]

# view/terminal_view.py
import os
import sys
import time
from datetime import datetime
from core.map import Map
from core.unit import Unit, Knight, Pikeman, Crossbowman, LongSwordsman, EliteSkirmisher, CavalryArcher, Onager, LightCavalry, Scorpion, CappedRam, Trebuchet, EliteWarElephant, Monk, Castle, Wonder
from core.army import Army

# Essayer d'importer curses
try:
    import curses
    CURSES_AVAILABLE = True
except ImportError:
    CURSES_AVAILABLE = False
    # Fallback pour Windows (windows-curses)
    try:
        import windows_curses as curses
        CURSES_AVAILABLE = True
    except ImportError:
        CURSES_AVAILABLE = False

# Import conditionnel pour la capture clavier selon l'OS
if os.name == 'nt':
    import msvcrt
else:
    import select
    import tty
    import termios


class TerminalView:
    """
    Gère l'affichage ASCII dans le terminal (Req 9.a).
    Avec curses pour un rendu lisse sans cligotement.
    Supporte les raccourcis clavier P (Pause), TAB (Snapshot HTML).
    """
    
    # Codes couleurs ANSI (fallback si curses non disponible)
    BLUE = '\033[94m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    GREY = '\033[90m'
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    
    # Symboles
    SYMBOL_EMPTY = '.'
    SYMBOL_TREE = '♣'
    SYMBOL_ROCK = '▲'
    
    def __init__(self, map_instance: Map):
        self.map = map_instance
        self.width = int(map_instance.width)
        self.height = int(map_instance.height)
        # Limiter l'affichage pour les grandes cartes
        self.max_display_width = min(80, self.width)
        self.max_display_height = min(25, self.height)
        # Offset pour le scroll (ZQSD)
        self.scroll_x = 0
        self.scroll_y = 0
        
        # Système de messages persistants
        self.message_queue: list[tuple[str, int]] = []  # (message, frames_remaining)
        self.frame_count = 0
        
        # État curses
        self.use_curses = CURSES_AVAILABLE
        self.stdscr = None
        
        # Initialiser curses si disponible
        if self.use_curses:
            try:
                self.stdscr = curses.initscr()
                curses.cbreak()
                curses.noecho()
                self.stdscr.nodelay(True)  # Non-bloquant pour getch()
                self.stdscr.keypad(True)   # Activer les touches spéciales (flèches, F-keys)
                
                # Initialiser les couleurs
                if curses.has_colors():
                    curses.start_color()
                    # Couleur 1: Bleu
                    curses.init_pair(1, curses.COLOR_BLUE, curses.COLOR_BLACK)
                    # Couleur 2: Rouge
                    curses.init_pair(2, curses.COLOR_RED, curses.COLOR_BLACK)
                    # Couleur 3: Vert
                    curses.init_pair(3, curses.COLOR_GREEN, curses.COLOR_BLACK)
                    # Couleur 4: Jaune
                    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
                    # Couleur 5: Gris (blanc dim)
                    curses.init_pair(5, curses.COLOR_WHITE, curses.COLOR_BLACK)
            except Exception as e:
                print(f"Erreur initialisation curses: {e}")
                self.use_curses = False
                self._restore_terminal()
    
    def _restore_terminal(self):
        """Restaure l'état du terminal."""
        try:
            if self.stdscr:
                self.stdscr.keypad(False)
                curses.echo()
                curses.nocbreak()
                curses.endwin()
                self.stdscr = None
        except:
            pass
    
    def cleanup(self):
        """Ferme proprement la vue terminal (appelé lors du switch de vue)."""
        self._restore_terminal()
    
    def __del__(self):
        """Nettoyer curses à la destruction."""
        self._restore_terminal()
    
    def _check_keyboard(self) -> str | None:
        """
        Vérifie si une touche a été pressée (non-bloquant).
        Retourne le caractère pressé ou None.
        """
        if self.use_curses and self.stdscr:
            try:
                key = self.stdscr.getch()
                if key == -1:  # Pas de touche
                    return None
                # Gestion des touches spéciales
                if key == curses.KEY_UP:
                    return 'up'
                elif key == curses.KEY_DOWN:
                    return 'down'
                elif key == curses.KEY_LEFT:
                    return 'left'
                elif key == curses.KEY_RIGHT:
                    return 'right'
                elif key == 9:  # TAB
                    return 'tab'
                elif key == 27:  # Échap
                    return 'escape'
                elif hasattr(curses, 'KEY_F9') and key == curses.KEY_F9:
                    return 'F9'
                elif hasattr(curses, 'KEY_F11') and key == curses.KEY_F11:
                    return 'F11'
                elif hasattr(curses, 'KEY_F12') and key == curses.KEY_F12:
                    return 'F12'
                elif key >= 32 and key < 127:
                    return chr(key).lower()
                else:
                    return None
            except Exception as e:
                return None
        else:
            # Fallback vers le système original
            if os.name == 'nt':
                # Windows
                if msvcrt.kbhit():
                    key = msvcrt.getch()
                    # Gestion des touches spéciales (F9, F11, F12, Flèches, Échap)
                    if key == b'\x00' or key == b'\xe0':
                        special = msvcrt.getch()
                        if special == b'C':  # F9
                            return 'F9'
                        elif special == b'\x85':  # F11
                            return 'F11'
                        elif special == b'\x86':  # F12
                            return 'F12'
                        elif special == b'H':  # Flèche haut
                            return 'up'
                        elif special == b'P':  # Flèche bas
                            return 'down'
                        elif special == b'M':  # Flèche droite
                            return 'right'
                        elif special == b'K':  # Flèche gauche
                            return 'left'
                    elif key == b'\x1b':  # Échap
                        return 'escape'
                    return key.decode('utf-8', errors='ignore').lower()
            else:
                # Linux/Mac
                if select.select([sys.stdin], [], [], 0)[0]:
                    return sys.stdin.read(1).lower()
        return None

    def display(self, armies: list[Army], time_elapsed: float, paused: bool = False, speed_multiplier: float = 1.0) -> str | None:
        """
        Génère et affiche la frame courante avec curses (ou fallback ANSI).
        Retourne une commande si une touche spéciale est pressée.
        """
        # Incrémenter le compteur de frames
        self.frame_count += 1
        
        # Vérifier les entrées clavier
        key = self._check_keyboard()
        command = None
        scroll_speed = 1
        
        if key == 'p':
            command = "toggle_pause"
        elif key == '\t' or key == 'tab':
            self._generate_html_snapshot(armies, int(time_elapsed))
            if not self.use_curses:
                print(f"{self.YELLOW}>>> Snapshot HTML généré et ouvert !{self.RESET}")
        elif key == 'escape':
            command = "quit"
        # Scrolling avec ZQSD
        elif key == 'z':
            self.scroll_y = max(0, self.scroll_y - scroll_speed)
        elif key == 's':
            self.scroll_y = min(self.height - self.max_display_height, self.scroll_y + scroll_speed)
        elif key == 'q':
            self.scroll_x = max(0, self.scroll_x - scroll_speed)
        elif key == 'd':
            self.scroll_x = min(self.width - self.max_display_width, self.scroll_x + scroll_speed)
        # Scrolling avec touches directionnelles
        elif key == 'up':
            self.scroll_y = max(0, self.scroll_y - scroll_speed)
        elif key == 'down':
            self.scroll_y = min(self.height - self.max_display_height, self.scroll_y + scroll_speed)
        elif key == 'left':
            self.scroll_x = max(0, self.scroll_x - scroll_speed)
        elif key == 'right':
            self.scroll_x = min(self.width - self.max_display_width, self.scroll_x + scroll_speed)
        # Commandes spéciales
        elif key == 'F9':
            command = "switch_view"
        elif key == 'F11':
            command = "quick_save"
        elif key == 'F12':
            command = "quick_load"
        
        # Mettre à jour le compteur de messages persistants
        self.message_queue = [(msg, frames - 1) for msg, frames in self.message_queue if frames > 1]
        
        # Rendu avec curses ou fallback
        if self.use_curses and self.stdscr:
            self._render_with_curses(armies, time_elapsed, paused, speed_multiplier)
        else:
            self._render_fallback(armies, time_elapsed, paused, speed_multiplier)
        
        time.sleep(0.033)  # ~30 FPS
        
        return command
    
    def _render_with_curses(self, armies: list[Army], time_elapsed: float, paused: bool, speed_multiplier: float):
        """Rendu avec curses pour un affichage lisse."""
        try:
            self.stdscr.clear()
            term_height, term_width = self.stdscr.getmaxyx()
            
            # Adapter dynamiquement les dimensions d'affichage à la taille du terminal
            # Réserver 7 lignes pour: header(1) + controles(1) + position(1) + vide(1) + bordure_haut(1) + bordure_bas(1) + stats(1)
            available_height = max(1, term_height - 7)
            available_width = max(1, term_width - 2)  # -2 pour les bordures gauche et droite
            
            # Mettre à jour les dimensions d'affichage max pour ce frame
            self.max_display_width = min(available_width, self.width)
            self.max_display_height = min(available_height, self.height)
            
            # Corriger le scroll si nécessaire (après redimensionnement)
            self.scroll_x = max(0, min(self.scroll_x, self.width - self.max_display_width))
            self.scroll_y = max(0, min(self.scroll_y, self.height - self.max_display_height))
            
            # Ligne 0: Header
            status = "PAUSE" if paused else "EN COURS"
            header = f"=== TEMPS {time_elapsed:.1f}s === [{status}] === x{speed_multiplier:.1f} ==="
            self.stdscr.addstr(0, 0, header[:term_width-1], curses.A_BOLD)
            
            # Ligne 1: Contrôles
            controls = "[P] Pause | [TAB] Infos | [ZQSD/Fleches] Scroll | [F9] Vue | [Esc] Quitter"
            self.stdscr.addstr(1, 0, controls[:term_width-1])
            
            # Ligne 2: Position et dimensions
            pos_info = f"Position: ({self.scroll_x}, {self.scroll_y}) | Affichage: {self.max_display_width}x{self.max_display_height}"
            self.stdscr.addstr(2, 0, pos_info[:term_width-1])
            
            # Calculer les dimensions visibles pour ce frame
            visible_width = min(self.max_display_width, self.width - self.scroll_x)
            visible_height = min(self.max_display_height, self.height - self.scroll_y)
            
            map_start_y = 4
            
            # Afficher la grille
            grid = [[self.SYMBOL_EMPTY for _ in range(visible_width)] for _ in range(visible_height)]
            grid_colors = [[0 for _ in range(visible_width)] for _ in range(visible_height)]  # 0 = aucune couleur
            
            # Obstacles
            for type_name, x, y in self.map.obstacles:
                ix, iy = int(x) - self.scroll_x, int(y) - self.scroll_y
                if 0 <= ix < visible_width and 0 <= iy < visible_height:
                    if type_name == "Tree":
                        grid[iy][ix] = self.SYMBOL_TREE
                        grid_colors[iy][ix] = 3  # Couleur verte
                    elif type_name == "Rock":
                        grid[iy][ix] = self.SYMBOL_ROCK
                        grid_colors[iy][ix] = 4  # Couleur grise
            
            # Unités
            for army in armies:
                color_pair = 1 if army.army_id == 0 else 2  # Bleu ou Rouge
                for unit in army.units:
                    if not unit.is_alive:
                        continue
                    ix = int(unit.pos[0]) - self.scroll_x
                    iy = int(unit.pos[1]) - self.scroll_y
                    if 0 <= ix < visible_width and 0 <= iy < visible_height:
                        symbol = self._get_unit_symbol(unit)
                        grid[iy][ix] = symbol
                        grid_colors[iy][ix] = color_pair
            
            # Afficher la bordure et la grille
            self.stdscr.addstr(map_start_y, 0, "+" + "-" * visible_width + "+")
            for iy, row in enumerate(grid):
                line_str = "|"
                for ix, symbol in enumerate(row):
                    color_pair = grid_colors[iy][ix]
                    if color_pair > 0:
                        try:
                            self.stdscr.addstr(map_start_y + 1 + iy, ix + 1, symbol, curses.color_pair(color_pair))
                        except:
                            self.stdscr.addstr(map_start_y + 1 + iy, ix + 1, symbol)
                    else:
                        try:
                            self.stdscr.addstr(map_start_y + 1 + iy, ix + 1, symbol)
                        except:
                            pass
                self.stdscr.addstr(map_start_y + 1 + iy, visible_width + 1, "|")
            
            # Stats
            stats_y = map_start_y + len(grid) + 3
            gen1 = armies[0].general.__class__.__name__
            gen2 = armies[1].general.__class__.__name__
            alive1 = self._count_alive(armies[0])
            total1 = len(armies[0].units)
            pct1 = (alive1 / total1 * 100) if total1 > 0 else 0
            alive2 = self._count_alive(armies[1])
            total2 = len(armies[1].units)
            pct2 = (alive2 / total2 * 100) if total2 > 0 else 0
            
            stats = f"Armée 1 [{gen1}]: {alive1}/{total1} ({pct1:.0f}%) | Armée 2 [{gen2}]: {alive2}/{total2} ({pct2:.0f}%)"
            self.stdscr.addstr(stats_y, 0, stats[:term_width-1])
            
            # Messages persistants
            msg_y = stats_y + 1
            for msg, _ in self.message_queue:
                if msg_y < term_height - 1:
                    try:
                        msg_str = str(msg)
                        msg_display = msg_str[:term_width-1] if len(msg_str) <= term_width-1 else msg_str[:term_width-2]
                        self.stdscr.addstr(msg_y, 0, msg_display)
                    except:
                        pass
                    msg_y += 1
            
            self.stdscr.refresh()
        except Exception as e:
            pass  # Silencieusement ignorer les erreurs de rendu curses
    
    def _render_fallback(self, armies: list[Army], time_elapsed: float, paused: bool, speed_multiplier: float):
        """Fallback vers le rendu ANSI si curses indisponible."""
        os.system('cls' if os.name == 'nt' else 'clear')
        
        status = f"{self.RED}[⏸ PAUSE]{self.RESET}" if paused else f"{self.GREEN}[▶ EN COURS]{self.RESET}"
        print(f"=== TEMPS {time_elapsed:.1f}s === {status} ===")
        print(f"Contrôles: [P] Pause | [TAB] Infos | [ZQSD/↑↓←→] Scroll | [F9] Changer vue | [Esc] Quitter")
        print(f"Position affichage: ({self.scroll_x}, {self.scroll_y})\n")
        
        visible_width = min(self.max_display_width, self.width - self.scroll_x)
        visible_height = min(self.max_display_height, self.height - self.scroll_y)
        
        grid = [[self.SYMBOL_EMPTY for _ in range(visible_width)] for _ in range(visible_height)]
        
        for type_name, x, y in self.map.obstacles:
            ix, iy = int(x) - self.scroll_x, int(y) - self.scroll_y
            if 0 <= ix < visible_width and 0 <= iy < visible_height:
                if type_name == "Tree":
                    grid[iy][ix] = f"{self.GREEN}{self.SYMBOL_TREE}{self.RESET}"
                elif type_name == "Rock":
                    grid[iy][ix] = f"{self.GREY}{self.SYMBOL_ROCK}{self.RESET}"

        for army in armies:
            color = self.BLUE if army.army_id == 0 else self.RED
            for unit in army.units:
                if not unit.is_alive:
                    continue
                ix = int(unit.pos[0]) - self.scroll_x
                iy = int(unit.pos[1]) - self.scroll_y
                if 0 <= ix < visible_width and 0 <= iy < visible_height:
                    symbol = self._get_unit_symbol(unit)
                    grid[iy][ix] = f"{color}{symbol}{self.RESET}"

        output_lines = []
        output_lines.append("+" + "-" * visible_width + "+")
        for row in grid:
            output_lines.append("|" + "".join(row) + "|")
        output_lines.append("+" + "-" * visible_width + "+")
        
        gen1 = armies[0].general.__class__.__name__
        gen2 = armies[1].general.__class__.__name__
        alive1 = self._count_alive(armies[0])
        total1 = len(armies[0].units)
        pct1 = (alive1 / total1 * 100) if total1 > 0 else 0
        alive2 = self._count_alive(armies[1])
        total2 = len(armies[1].units)
        pct2 = (alive2 / total2 * 100) if total2 > 0 else 0

        stats = f"Armée 1 [{gen1}] (Bleu): {alive1}/{total1} ({pct1:.0f}%) | "
        stats += f"Armée 2 [{gen2}] (Rouge): {alive2}/{total2} ({pct2:.0f}%)"
        output_lines.append(stats)
        
        if self.message_queue:
            output_lines.append("")
            for msg, _ in self.message_queue:
                output_lines.append(msg)
        
        print("\n".join(output_lines))
    
    def add_message(self, message: str, duration_frames: int = 120):
        """
        Ajoute un message persistant qui s'affiche pendant N frames.
        Par défaut 120 frames = ~4 secondes à 30 FPS.
        """
        self.message_queue.append((message, duration_frames))

    def _generate_html_snapshot(self, armies: list[Army], time_seconds: int):
        """
        Génère un fichier HTML instantané listant toutes les unités et leurs stats.
        Requis par le PDF (touche TAB).
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"saves/snapshot_time_{time_seconds}s_{timestamp}.html"
        
        os.makedirs("saves", exist_ok=True)
        
        html = [
            "<!DOCTYPE html>",
            "<html><head><meta charset='UTF-8'>",
            f"<title>MedievAIl - État du jeu à {time_seconds}s</title>",
            "<style>",
            "* { box-sizing: border-box; }",
            "body { font-family: 'Segoe UI', 'Arial', sans-serif; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #eee; padding: 20px; margin: 0; }",
            "container { max-width: 1400px; margin: 0 auto; }",
            "h1 { color: #e94560; text-shadow: 0 2px 4px rgba(0,0,0,0.5); border-bottom: 3px solid #e94560; padding-bottom: 10px; }",
            "h2 { margin-top: 30px; padding: 10px; border-radius: 5px; }",
            ".blue-team h2 { background: rgba(77, 157, 224, 0.2); color: #4d9de0; border-left: 4px solid #4d9de0; }",
            ".red-team h2 { background: rgba(233, 69, 96, 0.2); color: #e94560; border-left: 4px solid #e94560; }",
            "table { border-collapse: collapse; width: 100%; margin: 15px 0; background: #16213e; }",
            "th, td { border: 1px solid #444; padding: 10px; text-align: left; }",
            "th { background: #0f3460; font-weight: bold; color: #4d9de0; }",
            "tr:hover { background: #1a2d4d; }",
            ".dead { color: #888; text-decoration: line-through; background: #2a1a1a; }",
            ".alive { color: #4ade80; }",
            ".health-bar { display: inline-block; width: 100px; height: 15px; background: #333; border: 1px solid #555; border-radius: 3px; overflow: hidden; }",
            ".health-fill { height: 100%; background: linear-gradient(90deg, #ef4444 0%, #fbbf24 50%, #22c55e 100%); transition: width 0.2s; }",
            ".stats-summary { display: flex; gap: 20px; margin: 20px 0; flex-wrap: wrap; }",
            ".stat-box { background: #16213e; padding: 15px; border-radius: 5px; border-left: 4px solid #4d9de0; flex: 1; min-width: 200px; }",
            ".stat-box.red { border-left-color: #e94560; }",
            ".stat-box h3 { margin: 0 0 10px 0; }",
            ".stat-box p { margin: 5px 0; }",
            "footer { margin-top: 30px; border-top: 1px solid #444; padding-top: 20px; color: #888; text-align: center; }",
            "</style>",
            "</head><body>",
            "<div class='container'>",
            f"<h1>🏰 MedievAIl - État du jeu à {time_seconds}s</h1>",
            f"<p>Généré le {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>",
        ]
        
        for i, army in enumerate(armies):
            color_class = "blue-team" if i == 0 else "red-team"
            alive_count = self._count_alive(army)
            total_count = len(army.units)
            total_hp = sum(u.current_hp for u in army.units)
            max_hp = sum(u.max_hp for u in army.units)
            hp_percent = (total_hp / max_hp * 100) if max_hp > 0 else 0
            
            html.append(f"<div class='{color_class}'>")
            html.append(f"<h2>⚔️ Armée {i+1} - {army.general.__class__.__name__}</h2>")
            
            html.append("<div class='stats-summary'>")
            html.append(f"<div class='stat-box {'red' if i == 1 else ''}'>")
            html.append(f"<h3>Unités</h3>")
            html.append(f"<p><span class='alive'>{alive_count}</span> vivantes / <span>{total_count}</span> totales")
            html.append(f"<br><strong>{alive_count/total_count*100:.1f}%</strong> de survie")
            html.append("</p></div>")
            
            html.append(f"<div class='stat-box {'red' if i == 1 else ''}'>")
            html.append(f"<h3>Santé globale</h3>")
            html.append(f"<p><strong>{total_hp}/{max_hp} HP</strong> ({hp_percent:.1f}%)</p>")
            html.append(f"<div class='health-bar'><div class='health-fill' style='width: {hp_percent}%'></div></div>")
            html.append("</div></div>")
            
            html.append("<table>")
            html.append("<tr>")
            html.append("<th>Type d'unité</th>")
            html.append("<th>ID</th>")
            html.append("<th>Santé</th>")
            html.append("<th>Position</th>")
            html.append("<th>Attaque</th>")
            html.append("<th>Armure</th>")
            html.append("<th>Tâche actuelle</th>")
            html.append("<th>Statut</th>")
            html.append("</tr>")
            
            for unit in army.units:
                status_class = "dead" if not unit.is_alive else "alive"
                status_text = "🪦 Mort" if not unit.is_alive else "✓ Vivant"
                armor_text = f"Mêlée: {unit.melee_armor} | Perçant: {unit.pierce_armor}"
                hp_percent = (unit.current_hp / unit.max_hp * 100) if unit.max_hp > 0 else 0
                task = getattr(unit, 'statut', 'Inactif')
                
                html.append(f"<tr class='{status_class}'>")
                html.append(f"<td><strong>{unit.__class__.__name__}</strong></td>")
                html.append(f"<td>{unit.unit_id}</td>")
                html.append(f"<td>")
                html.append(f"<div class='health-bar'><div class='health-fill' style='width: {hp_percent}%'></div></div>")
                html.append(f"<small>{unit.current_hp}/{unit.max_hp}</small>")
                html.append(f"</td>")
                html.append(f"<td>({unit.pos[0]:.1f}, {unit.pos[1]:.1f})</td>")
                html.append(f"<td>{unit.attack_power} ({unit.attack_type})</td>")
                html.append(f"<td>{armor_text}</td>")
                html.append(f"<td><em>{task.capitalize()}</em></td>")
                html.append(f"<td>{status_text}</td>")
                html.append("</tr>")
            
            html.append("</table>")
            html.append("</div>")
        
        html.append("<footer>")
        html.append("<p>💡 Utilisez les contrôles du terminal pour explorer la carte en temps réel</p>")
        html.append("</footer>")
        html.append("</div></body></html>")
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("\n".join(html))
        
        try:
            if os.name == 'nt':
                os.startfile(os.path.abspath(filename))
            else:
                import subprocess
                subprocess.Popen(['xdg-open', os.path.abspath(filename)])
        except Exception as e:
            pass

    def _get_unit_symbol(self, unit: Unit) -> str:
        """Retourne une lettre selon le type d'unité (inspiré de AoE2).
        Majuscules pour armée 1 (bleue), minuscules pour armée 2 (rouge).
        """
        # Symboles de base (majuscules)
        if isinstance(unit, Knight): symbol = 'K'
        elif isinstance(unit, Pikeman): symbol = 'P'
        elif isinstance(unit, Crossbowman): symbol = 'X'
        elif isinstance(unit, LongSwordsman): symbol = 'S'
        elif isinstance(unit, EliteSkirmisher): symbol = 'E'
        elif isinstance(unit, CavalryArcher): symbol = 'A'
        elif isinstance(unit, Onager): symbol = 'O'
        elif isinstance(unit, LightCavalry): symbol = 'C'
        elif isinstance(unit, Scorpion): symbol = 'R'
        elif isinstance(unit, CappedRam): symbol = 'M'
        elif isinstance(unit, Trebuchet): symbol = 'T'
        elif isinstance(unit, EliteWarElephant): symbol = 'W'
        elif isinstance(unit, Monk): symbol = '✦'
        elif isinstance(unit, Castle): symbol = '■'
        elif isinstance(unit, Wonder): symbol = '◆'
        else: return '?'
        
        # Convertir en minuscule si armée 2 (army_id == 1)
        return symbol.lower() if hasattr(unit, 'army_id') and unit.army_id == 1 else symbol

    def _count_alive(self, army: Army) -> int:
        return sum(1 for u in army.units if u.is_alive)

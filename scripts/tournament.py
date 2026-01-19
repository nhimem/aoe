# scripts/tournament.py
"""
Tournoi automatique entre généraux.
Génère un rapport HTML complet avec matrices de scores.
"""
import sys
import os
import itertools
from collections import defaultdict
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.map import Map
from core.definitions import GENERAL_CLASS_MAP, UNIT_CLASS_MAP
from utils.unified_loader import load_scenario
from utils.loaders import load_map_from_file
from engine import Engine


class Tournament:
    """
    Execute un tournoi round-robin entre generaux sur plusieurs scenarios.
    
    Fonctionnalites:
    - Matchups entre generaux differents
    - Alternance des positions (joueur 0 / joueur 1) par defaut
    - Rapport HTML avec 4 matrices de scores
    """
    
    def __init__(self, general_names: list[str], scenario_paths: list[str], 
                 rounds: int = 10, alternate_positions: bool = True,
                 army_file: str = None):
        """
        Args:
            general_names: Liste des noms de generaux a combattre
            scenario_paths: Liste des chemins vers les scenarios (.scen, .map)
            rounds: Nombre de rounds par matchup
            alternate_positions: Si True, alterne les positions (P0/P1) sur les rounds
            army_file: Fichier armee a utiliser (defaut: 10 Knights)
        """
        self.general_names = general_names
        self.scenario_paths = scenario_paths
        self.rounds = rounds
        self.alternate_positions = alternate_positions
        self.army_file = army_file
        
        # Historique des matchs
        # Format: {"scenario": str, "gen_p0": str, "gen_p1": str, "winner": str|None}
        self.match_history: list[dict] = []
        
    def run(self):
        """
        Exécute le tournoi complet et génère le rapport.
        """
        n_gens = len(self.general_names)
        total_matchups = len(self.scenario_paths) * n_gens * (n_gens - 1) * self.rounds
        current = 0
        
        print(f"\nTotal de matchs a jouer: {total_matchups}")
        print("-" * 60)
        
        for scenario_path in self.scenario_paths:
            scenario_name = os.path.basename(scenario_path)
            print(f"\n[SCENARIO] {scenario_name}")
            
            # Toutes les paires sans reflexifs (A vs B, B vs A, mais pas A vs A)
            for gen1_name, gen2_name in itertools.permutations(self.general_names, 2):
                
                for round_num in range(self.rounds):
                    current += 1
                    
                    # Déterminer qui est player 0 / player 1
                    if self.alternate_positions:
                        # Alterner: rounds pairs = gen1 en P0, rounds impairs = gen1 en P1
                        if round_num % 2 == 0:
                            p0_name, p1_name = gen1_name, gen2_name
                        else:
                            p0_name, p1_name = gen2_name, gen1_name
                    else:
                        # Pas d'alternance: gen1 toujours en P0
                        p0_name, p1_name = gen1_name, gen2_name
                    
                    progress = f"[{current}/{total_matchups}]"
                    print(f"  {progress} {p0_name} (P0) vs {p1_name} (P1)... ", end="", flush=True)
                    
                    # Charger et exécuter le match
                    winner_name = self._run_match(scenario_path, p0_name, p1_name)
                    
                    # Enregistrer le résultat
                    self.match_history.append({
                        "scenario": scenario_path,
                        "gen_p0": p0_name,
                        "gen_p1": p1_name,
                        "winner": winner_name
                    })
                    
                    if winner_name:
                        print(f"-> Gagnant: {winner_name}")
                    else:
                        print("-> Egalite")
        
        print("\n" + "=" * 60)
        print("Tournoi termine!")
        print("=" * 60)
        
        self._print_console_summary()
        self._generate_html_report()
    
    def _run_match(self, scenario_path: str, p0_name: str, p1_name: str) -> str | None:
        """
        Execute un match unique et retourne le nom du gagnant (ou None pour egalite).
        """
        try:
            # Charger le scenario (sans prints)
            import sys
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = StringIO()  # Capturer les prints du loader
            
            try:
                if scenario_path.endswith('.scen'):
                    game_map, army1, army2 = load_scenario(scenario_path, p0_name, p1_name)
                elif scenario_path.endswith('.map'):
                    game_map = load_map_from_file(scenario_path)
                    if self.army_file:
                        # Utiliser le fichier armee specifie
                        army1, army2 = self._create_armies_from_file(game_map, p0_name, p1_name)
                    else:
                        # Armee par defaut (10 Knights)
                        army1, army2 = self._create_default_armies(game_map, p0_name, p1_name)
                else:
                    sys.stdout = old_stdout
                    return None
            finally:
                sys.stdout = old_stdout
            
            # Executer le match (headless, rapide, quiet)
            engine = Engine(game_map, army1, army2)
            engine.run_game(max_turns=5000, view=None, logic_speed=1, quiet=True)
            
            # Determiner le gagnant
            if engine.winner is not None:
                return p0_name if engine.winner == 0 else p1_name
            return None  # Egalite
            
        except Exception as e:
            return None
    
    def _create_default_armies(self, game_map: Map, p0_name: str, p1_name: str):
        """Cree des armees par defaut pour les fichiers .map"""
        from core.army import Army
        from core.unit import Knight
        
        gen0_cls = GENERAL_CLASS_MAP[p0_name]
        gen1_cls = GENERAL_CLASS_MAP[p1_name]
        
        # Positions proches du centre pour garantir un combat rapide
        w, h = game_map.width, game_map.height
        cx, cy = w // 2, h // 2
        
        units0 = []
        units1 = []
        
        # 10 Knights par camp, positionnes proches l'un de l'autre
        for i in range(10):
            # Armee 0: a gauche du centre
            units0.append(Knight(i, 0, (cx - 8 + (i % 5), cy - 2 + (i // 5))))
            # Armee 1: a droite du centre
            units1.append(Knight(10000 + i, 1, (cx + 4 + (i % 5), cy - 2 + (i // 5))))
        
        army0 = Army(0, units0, gen0_cls(0))
        army1 = Army(1, units1, gen1_cls(1))
        
        return army0, army1
    
    def _create_armies_from_file(self, game_map: Map, p0_name: str, p1_name: str):
        """Cree des armees a partir d'un fichier armee specifie"""
        from core.army import Army
        from utils.loaders import load_army_from_file
        
        gen0_cls = GENERAL_CLASS_MAP[p0_name]
        gen1_cls = GENERAL_CLASS_MAP[p1_name]
        
        # Charger la configuration d'armee depuis le fichier
        # On utilise le meme fichier pour les deux armees mais avec des generaux differents
        base_army = load_army_from_file(self.army_file, 0, p0_name)
        
        # Positions proches du centre
        w, h = game_map.width, game_map.height
        cx, cy = w // 2, h // 2
        
        units0 = []
        units1 = []
        
        # Recree les unites avec des positions centrees
        unit_types = [type(u) for u in base_army.units]
        n_units = len(unit_types)
        
        for i, unit_cls in enumerate(unit_types):
            # Armee 0: a gauche du centre
            row, col = i // 5, i % 5
            units0.append(unit_cls(i, 0, (cx - 8 + col, cy - 2 + row)))
            # Armee 1: a droite du centre
            units1.append(unit_cls(10000 + i, 1, (cx + 4 + col, cy - 2 + row)))
        
        army0 = Army(0, units0, gen0_cls(0))
        army1 = Army(1, units1, gen1_cls(1))
        
        return army0, army1
    
    def _print_console_summary(self):
        """Affiche un résumé dans la console."""
        print("\nResume Global:")
        print("-" * 40)
        
        stats = {g: {"wins": 0, "draws": 0, "losses": 0, "total": 0} 
                 for g in self.general_names}
        
        for m in self.match_history:
            p0, p1, winner = m["gen_p0"], m["gen_p1"], m["winner"]
            
            # Stats pour P0
            stats[p0]["total"] += 1
            if winner == p0:
                stats[p0]["wins"] += 1
            elif winner is None:
                stats[p0]["draws"] += 1
            else:
                stats[p0]["losses"] += 1
            
            # Stats pour P1
            stats[p1]["total"] += 1
            if winner == p1:
                stats[p1]["wins"] += 1
            elif winner is None:
                stats[p1]["draws"] += 1
            else:
                stats[p1]["losses"] += 1
        
        for g in self.general_names:
            s = stats[g]
            win_rate = (s["wins"] / s["total"] * 100) if s["total"] > 0 else 0
            print(f"  {g}: {s['wins']}W / {s['draws']}D / {s['losses']}L "
                  f"({win_rate:.1f}% victoires)")
    
    def _generate_html_report(self, filename: str = "tournament_report.html"):
        """Génère le rapport HTML complet."""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = ["""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MedievAIl - Rapport de Tournoi</title>
    <style>
        :root {
            --bg-dark: #1a1a2e;
            --bg-card: #16213e;
            --accent: #e94560;
            --accent2: #0f3460;
            --text: #eaeaea;
            --win: #4ade80;
            --lose: #f87171;
            --draw: #fbbf24;
        }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: var(--bg-dark);
            color: var(--text);
            margin: 0;
            padding: 20px;
            line-height: 1.6;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: var(--accent);
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .subtitle {
            text-align: center;
            color: #888;
            margin-bottom: 30px;
        }
        h2 {
            color: var(--accent);
            border-bottom: 2px solid var(--accent2);
            padding-bottom: 10px;
            margin-top: 40px;
        }
        h3 {
            color: #aaa;
            margin-top: 25px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
            background: var(--bg-card);
            border-radius: 8px;
            overflow: hidden;
        }
        th, td {
            padding: 12px 15px;
            text-align: center;
            border-bottom: 1px solid #333;
        }
        th {
            background: var(--accent2);
            color: var(--text);
            font-weight: 600;
        }
        tr:hover {
            background: rgba(233, 69, 96, 0.1);
        }
        .win { color: var(--win); font-weight: bold; }
        .lose { color: var(--lose); }
        .draw { color: var(--draw); }
        .high { background: rgba(74, 222, 128, 0.2); }
        .low { background: rgba(248, 113, 113, 0.2); }
        .config {
            background: var(--bg-card);
            padding: 15px 20px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        .config span {
            margin-right: 30px;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>MedievAIl - Rapport de Tournoi</h1>
"""]
        
        html.append(f'<p class="subtitle">Généré le {timestamp}</p>')
        
        # Configuration
        html.append('<div class="config">')
        html.append(f'<span><strong>Généraux:</strong> {", ".join(self.general_names)}</span>')
        html.append(f'<span><strong>Scénarios:</strong> {len(self.scenario_paths)}</span>')
        html.append(f'<span><strong>Rounds/matchup:</strong> {self.rounds}</span>')
        html.append(f'<span><strong>Alternance:</strong> {"Oui" if self.alternate_positions else "Non"}</span>')
        html.append('</div>')
        
        # === Section A: Score Global ===
        html.append('<h2>A. Score Global (% victoires)</h2>')
        html.append(self._html_global_table())
        
        # === Section B: Général vs Général (Global) ===
        html.append('<h2>B. Général vs Général (tous scénarios)</h2>')
        html.append(self._html_head_to_head_table())
        
        # === Section C: Par Scénario ===
        html.append('<h2>C. Matchups par Scénario</h2>')
        unique_scenarios = sorted(set(m["scenario"] for m in self.match_history))
        for scenario in unique_scenarios:
            scenario_name = os.path.basename(scenario)
            html.append(f'<h3>{scenario_name}</h3>')
            html.append(self._html_head_to_head_table(scenario_filter=scenario))
        
        # === Section D: Général vs Scénario ===
        html.append('<h2>D. Général vs Scénario (% victoires)</h2>')
        html.append(self._html_general_vs_scenario_table())
        
        html.append('</div></body></html>')
        
        # Écrire le fichier
        try:
            with open(filename, "w", encoding="utf-8") as f:
                f.write("\n".join(html))
            print(f"\nRapport genere: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"\nErreur generation rapport: {e}")
    
    def _html_global_table(self) -> str:
        """Génère le tableau de score global."""
        stats = {g: {"w": 0, "d": 0, "l": 0, "t": 0} for g in self.general_names}
        
        for m in self.match_history:
            p0, p1, winner = m["gen_p0"], m["gen_p1"], m["winner"]
            
            for p in [p0, p1]:
                stats[p]["t"] += 1
                if winner == p:
                    stats[p]["w"] += 1
                elif winner is None:
                    stats[p]["d"] += 1
                else:
                    stats[p]["l"] += 1
        
        rows = ['<table>',
                '<tr><th>Général</th><th>Victoires</th><th>Égalités</th>'
                '<th>Défaites</th><th>Total</th><th>% Victoires</th></tr>']
        
        # Trier par win rate
        sorted_gens = sorted(self.general_names, 
                            key=lambda g: stats[g]["w"] / max(stats[g]["t"], 1),
                            reverse=True)
        
        for g in sorted_gens:
            s = stats[g]
            rate = (s["w"] / s["t"] * 100) if s["t"] > 0 else 0
            cls = "high" if rate >= 50 else "low"
            rows.append(f'<tr class="{cls}">'
                       f'<td><strong>{g}</strong></td>'
                       f'<td class="win">{s["w"]}</td>'
                       f'<td class="draw">{s["d"]}</td>'
                       f'<td class="lose">{s["l"]}</td>'
                       f'<td>{s["t"]}</td>'
                       f'<td>{rate:.1f}%</td></tr>')
        
        rows.append('</table>')
        return "\n".join(rows)
    
    def _html_head_to_head_table(self, scenario_filter: str = None) -> str:
        """Génère une matrice général vs général."""
        
        # Filtrer les matchs si nécessaire
        matches = self.match_history
        if scenario_filter:
            matches = [m for m in matches if m["scenario"] == scenario_filter]
        
        # Calculer les stats head-to-head
        # matrix[g1][g2] = {"wins": x, "total": y} signifie g1 a gagné x fois contre g2
        matrix = defaultdict(lambda: defaultdict(lambda: {"w": 0, "t": 0}))
        
        for m in matches:
            p0, p1, winner = m["gen_p0"], m["gen_p1"], m["winner"]
            
            # P0 vs P1
            matrix[p0][p1]["t"] += 1
            if winner == p0:
                matrix[p0][p1]["w"] += 1
            
            # P1 vs P0 (symétrique)
            matrix[p1][p0]["t"] += 1
            if winner == p1:
                matrix[p1][p0]["w"] += 1
        
        # Construire le tableau
        rows = ['<table>', '<tr><th>vs</th>']
        for g in self.general_names:
            rows.append(f'<th>{g}</th>')
        rows.append('</tr>')
        
        for g1 in self.general_names:
            rows.append(f'<tr><th>{g1}</th>')
            for g2 in self.general_names:
                s = matrix[g1][g2]
                if s["t"] > 0:
                    rate = s["w"] / s["t"] * 100
                    cls = "win" if rate > 50 else ("lose" if rate < 50 else "draw")
                    cell = f'{s["w"]}/{s["t"]} ({rate:.0f}%)'
                    rows.append(f'<td class="{cls}">{cell}</td>')
                else:
                    rows.append('<td>-</td>')
            rows.append('</tr>')
        
        rows.append('</table>')
        return "\n".join(rows)
    
    def _html_general_vs_scenario_table(self) -> str:
        """Génère le tableau général vs scénario."""
        
        unique_scenarios = sorted(set(m["scenario"] for m in self.match_history))
        
        # Calculer les stats
        stats = defaultdict(lambda: defaultdict(lambda: {"w": 0, "t": 0}))
        
        for m in self.match_history:
            scenario = m["scenario"]
            p0, p1, winner = m["gen_p0"], m["gen_p1"], m["winner"]
            
            for p in [p0, p1]:
                stats[p][scenario]["t"] += 1
                if winner == p:
                    stats[p][scenario]["w"] += 1
        
        # Construire le tableau
        rows = ['<table>', '<tr><th>Général</th>']
        for scenario in unique_scenarios:
            rows.append(f'<th>{os.path.basename(scenario)}</th>')
        rows.append('</tr>')
        
        for g in self.general_names:
            rows.append(f'<tr><th>{g}</th>')
            for scenario in unique_scenarios:
                s = stats[g][scenario]
                if s["t"] > 0:
                    rate = s["w"] / s["t"] * 100
                    cls = "win" if rate >= 50 else "lose"
                    rows.append(f'<td class="{cls}">{rate:.1f}%</td>')
                else:
                    rows.append('<td>-</td>')
            rows.append('</tr>')
        
        rows.append('</table>')
        return "\n".join(rows)


# Point d'entrée pour tests directs
if __name__ == "__main__":
    # Test rapide
    tournament = Tournament(
        ["MajorDAFT", "ColonelKAISER"],
        ["maps/small.map"],
        rounds=2,
        alternate_positions=True
    )
    tournament.run()

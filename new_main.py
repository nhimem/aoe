# main.py
"""
MedievAIl - Battle GenerAIl Simulator
Point d'entrée CLI Principal.
"""
import argparse
import sys
import os
from typing import Optional
import importlib.util

# --- Import Core ---
from core.map import Map
from core.army import Army
from engine import Engine
from view.terminal_view import TerminalView
from view.gui_view import PygameView
from utils.serialization import save_game, load_game
from scripts.tournament import Tournament
from utils.loaders import load_map_from_file, load_army_from_file
from core.definitions import GENERAL_CLASS_MAP, UNIT_CLASS_MAP
from scripts.run_scenario import lanchester_scenario, custom_battle_scenario
from utils.generators import generate_map_file, generate_army_file


def main(args: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(
        prog="battle",
        description="MedievAIl - Battle GenerAIl Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # =========================================================================
    # 1. COMMAND: MATCH (Demo RL Match with Custom Args)
    # =========================================================================
    # Usage: python main.py match --map-size 150 --units 100 --maxturn -1
    match_parser = subparsers.add_parser("match", help="Run a GUI Demo Match using Trained RL Models")
    match_parser.add_argument("--map-size", type=int, default=120, help="Map dimension (e.g. 120)")
    match_parser.add_argument("--units", type=int, default=50, help="Number of units per team")
    match_parser.add_argument("--maxturn", type=int, default=2000, help="Max turns limit (-1 for infinite)")

    # =========================================================================
    # 2. COMMAND: TRAIN (Train RL Agent)
    # =========================================================================
    # Usage: python main.py train --episodes 1000 --map-size 80
    train_parser = subparsers.add_parser("train", help="Train the RL Agents")
    train_parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
    train_parser.add_argument("--map-size", type=int, default=80, help="Training map size")
    train_parser.add_argument("--units", type=int, default=40, help="Units per team for training")

    # =========================================================================
    # 3. EXISTING COMMANDS (Play, Run, Tourney, etc.)
    # =========================================================================
    play_parser = subparsers.add_parser("play", help="Quick Play (Legacy)")
    play_parser.add_argument("-t", "--terminal", action="store_true", help="Terminal mode")
    play_parser.add_argument("-u", "--units", nargs='+', default=["Knight"])
    play_parser.add_argument("-n", "--count", type=int, default=10)
    play_parser.add_argument("-ai", "--generals", nargs=2, default=["MajorDAFT", "MajorDAFT"])

    run_parser = subparsers.add_parser("run", help="Run Scenario")
    run_parser.add_argument("scenario", type=str)
    run_parser.add_argument("AI1", type=str)
    run_parser.add_argument("AI2", type=str)
    run_parser.add_argument("-t", "--terminal", action="store_true")

    tourney_parser = subparsers.add_parser("tourney", help="Run Tournament")
    # ... (giữ nguyên các tham số khác nếu cần)

    # PARSE ARGUMENTS
    parsed_args = parser.parse_args(args)

    # DISPATCH COMMANDS
    if parsed_args.command == "match":
        # Import lazy để tránh circular import nếu không cần thiết
        from run_rl_match import run_gui_match
        run_gui_match(
            map_size=parsed_args.map_size,
            units_per_team=parsed_args.units,
            max_turns=parsed_args.maxturn
        )

    elif parsed_args.command == "train":
        from rl_modules.trainer import train_agent
        train_agent(
            num_episodes=parsed_args.episodes,
            map_size=parsed_args.map_size,
            units_per_team=parsed_args.units
        )

    elif parsed_args.command == "play":
        run_play(parsed_args)
    elif parsed_args.command == "run":
        run_battle(parsed_args)
    elif parsed_args.command == "tourney":
        print("Feature not fully implemented in this merged version.")
    else:
        parser.print_help()


# --- Helper Functions for Legacy Commands (Giữ lại logic cũ) ---
def run_play(args):
    # (Giữ nguyên logic cũ của run_play từ file main.py gốc của bạn)
    print(f"Starting Quick Play: {args.count} {args.units} (AI: {args.generals})")
    game_map = Map(120, 120)
    # ... logic tạo army cũ ...
    # Để đơn giản, ở đây tôi gọi custom_battle_scenario
    gen1 = GENERAL_CLASS_MAP.get(args.generals[0], GENERAL_CLASS_MAP['MajorDAFT'])
    gen2 = GENERAL_CLASS_MAP.get(args.generals[1], GENERAL_CLASS_MAP['MajorDAFT'])

    comp = {u: args.count for u in (args.units if isinstance(args.units, list) else [args.units])}
    army1, army2 = custom_battle_scenario(comp, comp, gen1, gen2, (120, 120))

    engine = Engine(game_map, army1, army2)
    view = TerminalView(game_map) if args.terminal else PygameView(game_map, [army1, army2])
    engine.run_game(view=view)


def run_battle(args):
    # (Giữ nguyên logic cũ)
    print("Running Scenario...")
    # ... logic load scenario ...


if __name__ == "__main__":
    main()
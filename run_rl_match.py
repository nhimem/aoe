import sys
import os
import pickle
import pygame
import time

# Thêm đường dẫn root
sys.path.append(os.getcwd())

from engine import Engine
from core.army import Army
from extensions.map_builder import create_battle_map, generate_army_composition
from extensions.custom_view import CustomPygameView
from rl_modules.commander import RLCommander
from extensions.custom_units import GameCastle

MODEL_DIR = "models"
REPORT_DIR = "reports"


def load_trained_model(team_id):
    """Hàm hỗ trợ load Q-Table từ file .pkl"""
    filename = f"{MODEL_DIR}/q_table_team{team_id}_final.pkl"
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f:
                print(f">>> [LOAD] Đang nạp model cho Team {team_id} từ {filename}...")
                return pickle.load(f)
        except Exception as e:
            print(f">>> [LỖI] Không đọc được file model: {e}")
    return {}


# --- CUSTOM ENGINE: GHI ĐÈ LUẬT THẮNG ---
class RegicideEngine(Engine):
    def _check_game_over(self) -> bool:
        # Kiểm tra Castle
        castle_0_alive = any(isinstance(u, GameCastle) and u.is_alive for u in self.armies[0].units)
        castle_1_alive = any(isinstance(u, GameCastle) and u.is_alive for u in self.armies[1].units)

        # Kiểm tra Lính
        any_0_alive = any(u.is_alive for u in self.armies[0].units)
        any_1_alive = any(u.is_alive for u in self.armies[1].units)

        if not castle_0_alive:
            self.winner = 1
            self.game_over = True
            print(">>> Team 1 mất Castle! Team 2 thắng!")
            return True

        if not castle_1_alive:
            self.winner = 0
            self.game_over = True
            print(">>> Team 2 mất Castle! Team 1 thắng!")
            return True

        if not any_0_alive:
            self.winner = 1
            self.game_over = True
            print(">>> Team 1 bị tiêu diệt hoàn toàn! Team 2 thắng!")
            return True

        if not any_1_alive:
            self.winner = 0
            self.game_over = True
            print(">>> Team 2 bị tiêu diệt hoàn toàn! Team 1 thắng!")
            return True

        return False


def generate_battle_report(engine, army1, army2, winner):
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{REPORT_DIR}/match_report_{timestamp}.html"

    winner_text = "DRAW"
    if winner == 0:
        winner_text = "TEAM 1 (BLUE) WINS"
    elif winner == 1:
        winner_text = "TEAM 2 (RED) WINS"

    u1_alive = sum(1 for u in army1.units if u.is_alive)
    u2_alive = sum(1 for u in army2.units if u.is_alive)

    html_content = f"""
    <html>
    <head>
        <title>AOE Battle Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; background-color: #f4f4f4; padding: 20px; }}
            .container {{ background: white; padding: 20px; border-radius: 8px; max-width: 600px; margin: auto; }}
            h1 {{ color: #333; text-align: center; }}
            .result {{ font-size: 24px; font-weight: bold; text-align: center; padding: 10px; color: white; }}
            .win-blue {{ background-color: #3498db; }}
            .win-red {{ background-color: #e74c3c; }}
            .draw {{ background-color: #95a5a6; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Battle Report</h1>
            <div class="result {'win-blue' if winner == 0 else 'win-red' if winner == 1 else 'draw'}">
                {winner_text}
            </div>
            <table>
                <tr><th>Metric</th><th>Team 1</th><th>Team 2</th></tr>
                <tr><td>Alive</td><td>{u1_alive}</td><td>{u2_alive}</td></tr>
                <tr><td>Dead</td><td>{len(army1.units) - u1_alive}</td><td>{len(army2.units) - u2_alive}</td></tr>
            </table>
            <p style="text-align:center; margin-top:20px; color:#777;">Turns: {engine.turn_count}</p>
        </div>
    </body>
    </html>
    """
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n>>> 📝 Report saved: {filename}")


# --- HÀM CHÍNH ĐÃ CẬP NHẬT THAM SỐ ---
def run_gui_match(map_size=120, units_per_team=50, max_turns=2000):
    print(f"\n>>> KHỞI TẠO TRẬN ĐẤU DEMO (Match Mode)")
    print(
        f"    Map: {map_size}x{map_size} | Units: {units_per_team} | Max Turns: {max_turns if max_turns != -1 else 'INFINITE'}")

    # 1. Init Map
    game_map, tree_units = create_battle_map(width=map_size, height=map_size)

    # 2. Setup AI (Load Model)
    ai_1 = RLCommander(army_id=0, role_config="team1", learning=False)
    ai_1.q_table = load_trained_model(1)

    ai_2 = RLCommander(army_id=1, role_config="team2", learning=False)
    ai_2.q_table = load_trained_model(2)

    # 3. Spawn Armies (Tính toán vị trí spawn dựa trên size map)
    # Spawn cách mép 10-15 ô
    margin = 15
    spawn_1 = (margin, margin)
    spawn_2 = (map_size - margin, map_size - margin)

    units_1 = generate_army_composition(0, spawn_1[0], spawn_1[1], units_per_team)
    army_1 = Army(0, units_1, ai_1)

    units_2 = generate_army_composition(1, spawn_2[0], spawn_2[1], units_per_team)
    army_2 = Army(1, units_2, ai_2)

    # 4. Engine & View
    engine = RegicideEngine(game_map, army_1, army_2)
    view = CustomPygameView(game_map, engine.armies)
    view.set_nature_units(tree_units)  # Vẽ cây

    # 5. Xử lý max_turns
    if max_turns == -1:
        run_turns = sys.maxsize  # Vô hạn
    else:
        run_turns = max_turns

    print("\n>>> BẮT ĐẦU TRẬN ĐẤU...")
    print(">>> Phím SPACE: Pause | S: Step | +/-: Speed")

    try:
        engine.run_game(max_turns=run_turns, view=view, logic_speed=2)
    except KeyboardInterrupt:
        print("\n>>> Dừng trận đấu.")

    # 6. Báo cáo
    generate_battle_report(engine, army_1, army_2, engine.winner)


if __name__ == "__main__":
    # Chạy mặc định nếu gọi trực tiếp
    run_gui_match()
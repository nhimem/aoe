import os
import sys
import pickle
import matplotlib.pyplot as plt
from collections import deque

sys.path.append(os.getcwd())

from engine import Engine
from core.army import Army
from extensions.map_builder import create_battle_map, generate_army_composition
from extensions.custom_units import GameCastle
from rl_modules.commander import RLCommander

NUM_EPISODES = 500
MAX_TURNS = 2000
SAVE_INTERVAL = 50
MODEL_DIR = "models"
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY = 0.995


# Engine Custom cho Training
class RegicideEngine(Engine):
    def check_game_over(self):
        c1 = any(isinstance(u, GameCastle) and u.is_alive for u in self.army1.units)
        if not c1: return 1
        c2 = any(isinstance(u, GameCastle) and u.is_alive for u in self.army2.units)
        if not c2: return 0
        return super().check_game_over()


def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)


def save_q_table(q_table, filename):
    with open(filename, 'wb') as f: pickle.dump(q_table, f)


def train_agent():
    ensure_dir(MODEL_DIR)
    q_table_team1 = {}
    q_table_team2 = {}
    recent_wins = deque(maxlen=50)
    win_history = []
    epsilon = EPSILON_START

    print("TRAINING STARTED (Regicide Mode)")

    for episode in range(1, NUM_EPISODES + 1):
        # Lấy map, bỏ qua danh sách cây (Trainer không cần vẽ cây)
        game_map, _ = create_battle_map(width=80, height=80)

        ai_1 = RLCommander(0, "team1", learning=True)
        ai_2 = RLCommander(1, "team2", learning=True)
        ai_1.q_table = q_table_team1
        ai_2.q_table = q_table_team2
        ai_1.epsilon = ai_2.epsilon = epsilon

        army_1 = Army(0, generate_army_composition(0, 10, 10, 40), ai_1)
        army_2 = Army(1, generate_army_composition(1, 65, 65, 40), ai_2)

        # Engine không chứa cây -> Cây không phải Unit -> Không tính vào stats/win-loss
        engine = RegicideEngine(game_map, army_1, army_2)
        engine.run_game(max_turns=MAX_TURNS, logic_speed=10, quiet=True)

        # Reward Logic (Khuyến khích thắng)
        winner = engine.winner
        REWARD_WIN = 5000
        REWARD_LOSS = -2000
        REWARD_DRAW = -1000

        if winner == 0:
            recent_wins.append(1)
            res = "T1 WIN"
            ai_1.learn_terminal_result(REWARD_WIN)
            ai_2.learn_terminal_result(REWARD_LOSS)
        elif winner == 1:
            recent_wins.append(0)
            res = "T2 WIN"
            ai_1.learn_terminal_result(REWARD_LOSS)
            ai_2.learn_terminal_result(REWARD_WIN)
        else:
            recent_wins.append(0)
            res = "DRAW"
            ai_1.learn_terminal_result(REWARD_DRAW)
            ai_2.learn_terminal_result(REWARD_DRAW)

        win_rate = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
        win_history.append(win_rate)

        if epsilon > EPSILON_END: epsilon *= EPSILON_DECAY

        print(f"Ep {episode:03d} | Eps {epsilon:.2f} | {res} | WR(T1): {win_rate:.1f}%")

        if episode % SAVE_INTERVAL == 0:
            save_q_table(q_table_team1, f"{MODEL_DIR}/q_table_team1_ep{episode}.pkl")
            save_q_table(q_table_team2, f"{MODEL_DIR}/q_table_team2_ep{episode}.pkl")

    save_q_table(q_table_team1, f"{MODEL_DIR}/q_table_team1_final.pkl")
    save_q_table(q_table_team2, f"{MODEL_DIR}/q_table_team2_final.pkl")
    print("DONE.")
    try:
        plt.plot(win_history)
        plt.savefig(f"{MODEL_DIR}/training_chart.png")
    except:
        pass


if __name__ == "__main__":
    train_agent()
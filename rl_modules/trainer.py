
import os
import sys 
import pickle #thư viện dùng để lưu các object Python (list, dict_ ra file nhị phân => dùng để lưu não AI (q_table)
import matplotlib.pyplot as plt
from collections import deque #1 loại list đặc biệt, tối ưu cho việc thêm/xoá 2 đầu. Lưu lịch sử thắng/thua gần đây

sys.path.append(os.getcwd()) #tìm file code ở thư mục hiện tại, giúp python tìm thấy các module trong thư mục project
#Import các module game!
from engine import Engine
from core.army import Army
from extensions.map_builder import create_battle_map, generate_army_composition
# Import đúng kiến trúc cũ
from extensions.custom_units import GameCastle
from rl_modules.commander import RLCommander

# Các hằng số mặc định
NUM_EPISODES = 500      #tổng số trận đấu sẽ tập luyện
MAX_TURNS = 2000        #Giới hạn số lượt đi mỗi trận (tránh trận đấu kéo dài vô hạn)
SAVE_INTERVAL = 50      #Cứ 50 trận thì lưu file 1 lần 
MODEL_DIR = "ai/rl/models"    #Thư mục lưu file
EPSILON_START = 1.0     #Lúc đầu: 100% exploration
EPSILON_END = 0.05      #Lúc cuối chỉ còn 5% exploration
EPSILON_DECAY = 0.995   #exploration giảm sau mỗi trận. => trận 1: 100%, trận 2 = 1.0 x 0.995 = 0.995
#Cơ chế Annealing: từ "exploration" sang "exploitation"


# Engine Custom cho Training
class RegicideEngine(Engine): #Ám sát vua
    def check_game_over(self):
        #Override luật thắng thua ddeer AI tập trung bảo vệ/tấn công nhà chính
        #Kiểm tra xem Castle team1 còn sống k? 
        c1 = any(isinstance(u, GameCastle) and u.is_alive for u in self.army1.units)
        if not c1: return 1 #team1 mất Castle => Team 2 thắng (return 1)
        c2 = any(isinstance(u, GameCastle) and u.is_alive for u in self.army2.units)
        if not c2: return 0 #team2 mất Castle => Team1 thắng (return 0)
        return super().check_game_over() #Nếu cả 2: dùng luật cũ: hết lính 


def ensure_dir(directory):
    if not os.path.exists(directory): os.makedirs(directory)


def save_q_table(q_table, filename):
    with open(filename, 'wb') as f: pickle.dump(q_table, f)


# [ĐÃ SỬA] Hàm nhận tham số đầu vào từ main.py
def train_agent(num_episodes=NUM_EPISODES, map_size=80, units_per_team=40):
    ensure_dir(MODEL_DIR)
    q_table_team1 = {}
    q_table_team2 = {}
    recent_wins = deque(maxlen=50)
    win_history = []
    epsilon = EPSILON_START

    print(f"TRAINING STARTED (Regicide Mode) | Episodes: {num_episodes} | Map: {map_size}x{map_size} | Units: {units_per_team}")

    # Tính toán vị trí spawn dựa trên map_size (Margin 15 đơn vị)
    # Để tránh spawn ngoài bản đồ nếu map nhỏ
    margin = 15
    spawn_1 = (margin, margin)
    spawn_2 = (map_size - margin, map_size - margin)

    for episode in range(1, num_episodes + 1):
        # [THAM SỐ] Sử dụng map_size truyền vào
        game_map, _ = create_battle_map(width=map_size, height=map_size)

        ai_1 = RLCommander(0, "team1", learning=True)
        ai_2 = RLCommander(1, "team2", learning=True)
        ai_1.q_table = q_table_team1
        ai_2.q_table = q_table_team2
        ai_1.epsilon = ai_2.epsilon = epsilon

        # [THAM SỐ] Sử dụng units_per_team truyền vào
        army_1 = Army(0, generate_army_composition(0, spawn_1[0], spawn_1[1], units_per_team), ai_1)
        army_2 = Army(1, generate_army_composition(1, spawn_2[0], spawn_2[1], units_per_team), ai_2)

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
        plt.title(f"Training Progress (Map {map_size}, Units {units_per_team})")
        plt.savefig(f"{MODEL_DIR}/training_chart.png")
    except:
        pass


if __name__ == "__main__":
    train_agent()
"""
Frame của trận đấu: đơn vị thời gian nhỏ nhất mà game xử lý logic, gồm:
    Trong mỗi vòng lặp while của engine.run_game():
        - Quan sát: vị trí lính, máu, công trình còn hay mất
        + AI (RLCommander) nhìn vào trạng thái này, thông qua _get_state_key
        - Ra quyết định: hàm decision_actions của AI được gọi => AI tính toán và trả về danh sách lệnh: mov, attacks
        - Cập nhật Logic: Là việc của Engine. Thực thi các lệnh trên:
            + Toạ độ lính thay đổi
            + Máu lính thay đổi (nếu tấn công và trúng địch)
            + Lính chết (nếu máu về 0)
        - Kiểm tra kết thúc:
            + check_game_over: chạy để xem nhà chính/vua còn sống k ? Chết => endgame
        - Trả thưởng cho RL: tính điểm chênh lệch: reward = điểm mới - điểm cũ - 1
            + cập nhật bảng q_table
        -
"""

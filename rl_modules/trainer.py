
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
MODEL_DIR = "ai/rl/models"    #Thư mục lưu file chứa q-table
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


def ensure_dir(directory): #đảm bảo thư mục tồn tại, nếu chưa tồn tại thì tạo thư mục đó.
    #đảm bảo trước khi lưu 1 file nào đó, folder chứa nó phải có trên ổ cứng rồi.
    if not os.path.exists(directory): os.makedirs(directory)


def save_q_table(q_table, filename): #lưu trữ q-table ra ổ cứng dưới dạng file, để có thể dùng lại hoặc huấn luyện tiếp
    with open(filename, 'wb') as f: pickle.dump(q_table, f) #Dùng pickle để đóng gói q-table và đổ vài file f.


# [ĐÃ SỬA] Hàm nhận tham số đầu vào từ main.py
#VÒNG LẶP HUẤN LUYỆN: tham số: 500 trận, map 80x80, mỗi bên 40 lính.
def train_agent(num_episodes=NUM_EPISODES, map_size=80, units_per_team=40): 
    ensure_dir(MODEL_DIR) #để lưu file chứa q-table
    q_table_team1 = {} #Tạo q-table cho cả 2 team, 2 q-table này sẽ được điền dần các kinh nghiệm
    q_table_team2 = {}
    recent_wins = deque(maxlen=50) #List chỉ lưu 50 trận gần nhất. Tính win rate, 50 trận thắng bao nhiêu %
    win_history = [] #Lưu lịch sử thắng
    epsilon = EPSILON_START #đặt độ tò mò (exploration) ban đầu là 100%

    print(f"TRAINING STARTED (Regicide Mode) | Episodes: {num_episodes} | Map: {map_size}x{map_size} | Units: {units_per_team}")

    # Tính toán vị trí spawn dựa trên map_size (Margin 15 đơn vị)
    # Để tránh spawn ngoài bản đồ nếu map nhỏ
    margin = 15
    spawn_1 = (margin, margin) #Team 1 ở góc trên trái
    spawn_2 = (map_size - margin, map_size - margin) #Team2 ở góc dưới phải 
    #Lmaf như vậy giúp code chạy được với mọi kích thước map (80,100,200,..) mà không bị lỗi spawn(sinh ra) quân ra ngoài rìa

    #Vòng lặp huấn luyện: mỗi episode là 1 trận đấu trọn vẹn!
    for episode in range(1, num_episodes + 1):
        # [THAM SỐ] Sử dụng map_size truyền vào => Tạo bản đồ ngẫu nhiên => học cách đánh tổng quát, không thuộc lòng 1 map cố định
        game_map, _ = create_battle_map(width=map_size, height=map_size)

        ai_1 = RLCommander(0, "team1", learning=True) #tạo 2 tướng chỉ huy (RLCommander) cho trận này
        ai_2 = RLCommander(1, "team2", learning=True)
        ai_1.q_table = q_table_team1 #gán bộ não tổng vào cho từng tướng => Thân xác thì mới nhưng ký ức thì được nạp từ kho tổng hợp dữ liệu vào
        ai_2.q_table = q_table_team2
        ai_1.epsilon = ai_2.epsilon = epsilon #Cả 2 bên có độ tò mò như nhau

        # [THAM SỐ] Sử dụng units_per_team truyền vào
        #Tạo quân đội (lính, nhà) sử dụng vị trí spawn đã tính ở trên, giao quyền chỉ huy cho ai_1 và ai_2
        army_1 = Army(0, generate_army_composition(0, spawn_1[0], spawn_1[1], units_per_team), ai_1) 
        army_2 = Army(1, generate_army_composition(1, spawn_2[0], spawn_2[1], units_per_team), ai_2)

        # Engine không chứa cây -> Cây không phải Unit -> Không tính vào stats/win-loss
        #Khởi tạo bộ máy game RegicideEngine
        engine = RegicideEngine(game_map, army_1, army_2)
        engine.run_game(max_turns=MAX_TURNS, logic_speed=10, quiet=True) 
        #quiet = true => no graphics => giúp máy tính dồn sức tính toán logic, chạy nhanh hơn
        #Hàm chạy cho đến khi có người thắng hoặc hết 2000 turn (lượt). 
        #RLCommander liên tục cập nhật q-table qua từng bước nhỏ (step reward)

        # Reward Logic (Khuyến khích thắng)
        #Lấy kết quả và định nghĩa điểm thưởng
        #Điểm thắng rắt lớn => khuyến khích AI khao khát chiến thắng
        winner = engine.winner
        REWARD_WIN = 5000
        REWARD_LOSS = -2000 
        REWARD_DRAW = -1000

        #Nếu team1 thắng:
        if winner == 0:
            recent_wins.append(1) #đưa vào lịch sử thắng 
            res = "T1 WIN"
            ai_1.learn_terminal_result(REWARD_WIN) #team1 được +5000 điểm vào hành động cuối cùng
            ai_2.learn_terminal_result(REWARD_LOSS) #team2 bị trừ 2000đ
            
        elif winner == 1: #Tương tự cho trường hợp team2 thắng
            recent_wins.append(0)
            res = "T2 WIN"
            ai_1.learn_terminal_result(REWARD_LOSS)
            ai_2.learn_terminal_result(REWARD_WIN)
            
        else: #2 team hoà
            recent_wins.append(0)
            res = "DRAW"
            ai_1.learn_terminal_result(REWARD_DRAW)
            ai_2.learn_terminal_result(REWARD_DRAW)

        #Tính tỉ lệ thắng của team1 win cho tới thời điểm hiện tại
        win_rate = sum(recent_wins) / len(recent_wins) * 100 if recent_wins else 0
        win_history.append(win_rate)


        #Giảm epsilon. Sau mỗi trận, AI bớt exploration lại và tin vào kinh nghiệm bản thân nhiều hơn
        if epsilon > EPSILON_END: epsilon *= EPSILON_DECAY

        #In báo cáo ra màn hình để theo dõi tiến độ. Vd: Ep 100 | Eps 0.60 | T1 Win | WR(T1): 55.0%
        print(f"Ep {episode:03d} | Eps {epsilon:.2f} | {res} | WR(T1): {win_rate:.1f}%")

        #Cứ mỗi 50 trận, lưu q-table ra file 1 lần để backup (sao lưu - bản sao lưu , dự phòng)
        if episode % SAVE_INTERVAL == 0:
            save_q_table(q_table_team1, f"{MODEL_DIR}/q_table_team1_ep{episode}.pkl")
            save_q_table(q_table_team2, f"{MODEL_DIR}/q_table_team2_ep{episode}.pkl")


    #Kết thúc huấn luyện: khi vòng lặp chạy xong hết 500 trận
    #Lưu phiên bản cuối cùng. Phiên bản thông minh nhất
    save_q_table(q_table_team1, f"{MODEL_DIR}/q_table_team1_final.pkl")
    save_q_table(q_table_team2, f"{MODEL_DIR}/q_table_team2_final.pkl")
    print("DONE.").


    #Vẽ biểu đồ quá trình học và lưu thành ảnh: training_chart.pnt. Nhìn vào ảnh => biết AI có tiến bộ hay không?
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


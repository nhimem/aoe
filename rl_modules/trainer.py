
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
-----------------------------------------------------
run_rl_match.py: Chạy mô phỏng thực tế
trainer.py => chạy ngầm, khôgn hình ảnh, tốc độ cao để học
file này: dùng để biểu diễn kết quả học được ra màn hình đồ hoạ (GUI) cho con người xem, sau đó xuất ra 1 báo cáo HTML
-----------------------------------------------------

import sys
import os
import pickle #để đọc file q-table - bộ não của AI
import pygame #thư viện đồ hoạ để vẽ cửa sổ game
import time #để lấy thời gian thực tạo nên file báo cáo

# Thêm đường dẫn root
sys.path.append(os.getcwd())

#import các module tự viết trong project
from engine import Engine
from core.army import Army
from extensions.map_builder import create_battle_map, generate_army_composition
from extensions.custom_view import CustomPygameView
from rl_modules.commander import RLCommander
from extensions.custom_units import GameCastle

MODEL_DIR = "ai/rl/models" #thư mục chứa AI đã train, tức q-table
REPORT_DIR = "reports" #thư mục sẽ lưu file báo cáo HTML vào


#Nạp bộ não - q-table vào từ file đã lưu
def load_trained_model(team_id):
    """Hàm hỗ trợ load Q-Table từ file .pkl"""
    filename = f"{MODEL_DIR}/q_table_team{team_id}_final.pkl"
    if os.path.exists(filename):
        try:
            with open(filename, 'rb') as f: #Mở file chế độ đọc binary
                print(f">>> [LOAD] Đang nạp model cho Team {team_id} từ {filename}...")
                return pickle.load(f) #Load dữ liệu thành DICT
        except Exception as e:
            print(f">>> [LỖI] Không đọc được file model: {e}")
    return {} #Nếu lỗi không thấy file trả về não rỗng


# --- CUSTOM ENGINE: GHI ĐÈ (OVERRIDE) LUẬT THẮNG ---
class RegicideEngine(Engine):
    def _check_game_over(self) -> bool:
        # Kiểm tra Castle mỗi bên còn sống không
        # 0: team 1, 1: team 2
        
        castle_0_alive = any(isinstance(u, GameCastle) and u.is_alive for u in self.armies[0].units)
        castle_1_alive = any(isinstance(u, GameCastle) and u.is_alive for u in self.armies[1].units)

    
        # Kiểm tra Lính còn sống k
        any_0_alive = any(u.is_alive for u in self.armies[0].units)
        any_1_alive = any(u.is_alive for u in self.armies[1].units)

        if not castle_0_alive: #Team 0 mất nhà, team 1 thắng
            self.winner = 1
            self.game_over = True
            print(">>> Team 1 mất Castle! Team 2 thắng!")
            return True

        if not castle_1_alive: #Team 1 mất nhà, team 0 thắng
            self.winner = 0
            self.game_over = True
            print(">>> Team 2 mất Castle! Team 1 thắng!")
            return True

        if not any_0_alive: #team 0 không còn lính
            self.winner = 1
            self.game_over = True
            print(">>> Team 1 bị tiêu diệt hoàn toàn! Team 2 thắng!")
            return True
 
        if not any_1_alive: #team 1 không còn lính
            self.winner = 0
            self.game_over = True
            print(">>> Team 2 bị tiêu diệt hoàn toàn! Team 1 thắng!")
            return True

        return False


# [HÀM MỚI] Đếm số lượng quân ban đầu 
def get_initial_composition(army_units):
    """
    Hàm đếm số lượng quân ban đầu trước khi trận đấu diễn ra.
    Trả về dict: {'UnitName': count} Lưu vào 1 dict
    """
    comp = {}
    for u in army_units:
        u_type = type(u).__name__ #ví dụ "Archer", "Knight" => chính là tên của quân đó
        comp[u_type] = comp.get(u_type, 0) + 1
    return comp


# [HÀM MỚI] Đếm số lượng quân còn sống
def count_current_survivors(army_units):
    """
    Hàm đếm số lượng quân còn sống tại thời điểm gọi.
    Trả về dict: {'UnitName': count_alive}
    """
    alive = {}
    for u in army_units:
        if u.is_alive:
            u_type = type(u).__name__
            alive[u_type] = alive.get(u_type, 0) + 1
    return alive


def generate_unit_rows_html(initial_comp, survivor_comp):
    """
    Sinh HTML dựa trên so sánh giữa ban đầu và hiện tại.
    """
    html_rows = ""
    # Sắp xếp theo tên unit
    sorted_keys = sorted(initial_comp.keys())

    for u_type in sorted_keys:
        total = initial_comp[u_type] #Tổng ban đầu
        alive = survivor_comp.get(u_type, 0) #Còn sống
        dead = total - alive     #đã chết
        if dead < 0: dead = 0  # Đề phòng lỗi logic

        #Chèn vào template HTML
        html_rows += f"""
        <tr>
            <td class="sub-label">{u_type}</td>
            <td style="font-weight:bold; color:#555;">{total}</td>
            <td class="val-alive">{alive}</td>
            <td class="val-dead">{dead}</td>
        </tr>
        """
    return html_rows


#Tạo báo cáo tổng
def generate_battle_report(engine, winner, init_s1, init_s2, army1, army2):
    #Tạo thư mục reports nếu chưa có
    if not os.path.exists(REPORT_DIR):
        os.makedirs(REPORT_DIR)

    #Đặt tên theo file ngày giờ
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"{REPORT_DIR}/match_report_{timestamp}.html"

    # Xác định người thắng - xác định màu sắc và text cho winner
    winner_text = "DRAW"
    winner_bg = "#95a5a6"  # Gray
    if winner == 0:
        winner_text = "TEAM 1 (BLUE) WINS"
        winner_bg = "#3498db"  # Blue
    elif winner == 1:
        winner_text = "TEAM 2 (RED) WINS"
        winner_bg = "#e74c3c"  # Red

    #Tính toán số liệu hiện tại
    # Đếm số quân còn sống hiện tại
    current_s1 = count_current_survivors(army1.units)
    current_s2 = count_current_survivors(army2.units)

    # Tạo các dòng HTML
    rows_team1 = generate_unit_rows_html(init_s1, current_s1)
    rows_team2 = generate_unit_rows_html(init_s2, current_s2)

    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>MedievAIl Battle Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; background-color: #f4f6f8; color: #333; margin: 0; padding: 20px; }}
            .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .header {{ text-align: center; margin-bottom: 30px; border-bottom: 2px solid #eee; padding-bottom: 20px; }}
            .header h1 {{ margin: 0; color: #2c3e50; font-size: 32px; }}
            .header .meta {{ color: #7f8c8d; font-size: 14px; margin-top: 5px; }}
            .winner-banner {{ 
                background-color: {winner_bg}; color: white; 
                text-align: center; padding: 15px; font-size: 24px; font-weight: bold; 
                border-radius: 6px; margin-bottom: 30px; 
            }}
            .stats-container {{ display: flex; gap: 30px; }}
            .team-card {{ flex: 1; border: 1px solid #e1e4e8; border-radius: 8px; overflow: hidden; }}
            .team-header {{ padding: 15px; text-align: center; font-weight: bold; font-size: 18px; color: white; }}
            .team-blue {{ background-color: #3498db; }}
            .team-red {{ background-color: #e74c3c; }}
            .stat-table {{ width: 100%; border-collapse: collapse; }}
            .stat-table th, .stat-table td {{ padding: 10px 15px; border-bottom: 1px solid #eee; text-align: center; }}
            .stat-table th {{ background-color: #f8f9fa; font-size: 12px; text-transform: uppercase; color: #555; }}
            .sub-label {{ text-align: left !important; padding-left: 20px !important; font-weight: 500; color: #444; }}
            .val-alive {{ color: #27ae60; font-weight: bold; }}
            .val-dead {{ color: #c0392b; }}
            .footer {{ text-align: center; margin-top: 40px; font-size: 12px; color: #999; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Battle Report</h1>
                <div class="meta">Match ID: {timestamp} &bull; Duration: {engine.turn_count} Turns</div>
            </div>

            <div class="winner-banner">
                {winner_text}
            </div>

            <div class="stats-container">
                <div class="team-card">
                    <div class="team-header team-blue">Team 1 (Blue)</div>
                    <table class="stat-table">
                        <tr><th>Unit Type</th><th>Total</th><th>Alive</th><th>Dead</th></tr>
                        {rows_team1}
                    </table>
                </div>

                <div class="team-card">
                    <div class="team-header team-red">Team 2 (Red)</div>
                    <table class="stat-table">
                        <tr><th>Unit Type</th><th>Total</th><th>Alive</th><th>Dead</th></tr>
                        {rows_team2}
                    </table>
                </div>
            </div>

            <div class="footer">
                MedievAIl - RL Battle Simulator Report
            </div>
        </div>
    </body>
    </html>
    """
    #Ghi ra file
    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"\n>>> 📝 Report chi tiết đã lưu: {filename}")


# --- HÀM CHÍNH ---: điều khiển trận đấu
def run_gui_match(map_size=120, units_per_team=50, max_turns=2000):
    #Tạo map và cây cối
    print(f"\n>>> KHỞI TẠO TRẬN ĐẤU DEMO (Match Mode)")
    print(
        f"    Map: {map_size}x{map_size} | Units: {units_per_team} | Max Turns: {max_turns if max_turns != -1 else 'INFINITE'}")

    # 1. Init Map
    game_map, tree_units = create_battle_map(width=map_size, height=map_size)

    # 2. Setup AI (Load Model)
    #Learning = false => AI chỉ dùng kiến thức cũ để đánh, không học thêm
    ai_1 = RLCommander(army_id=0, role_config="team1", learning=False)
    ai_1.q_table = load_trained_model(1) #Load não

    ai_2 = RLCommander(army_id=1, role_config="team2", learning=False)
    ai_2.q_table = load_trained_model(2) #Load não

    # 3. Spawn Armies #Sinh quân
    margin = 15 #vị trí sinh quân
    spawn_1 = (margin, margin)
    spawn_2 = (map_size - margin, map_size - margin)

    units_1 = generate_army_composition(0, spawn_1[0], spawn_1[1], units_per_team)
    army_1 = Army(0, units_1, ai_1)

    units_2 = generate_army_composition(1, spawn_2[0], spawn_2[1], units_per_team)
    army_2 = Army(1, units_2, ai_2)

    # --- [GHI NHẬN SỐ LƯỢNG QUÂN BAN ĐẦU] ---
    init_stats_1 = get_initial_composition(units_1) #ghi lại số lượng quân lúc đầu
    init_stats_2 = get_initial_composition(units_2)
    print(f">>> Initial Stats T1: {init_stats_1}")
    print(f">>> Initial Stats T2: {init_stats_2}")

    # 4. Engine & View #Chạy engine với giao diện
    engine = RegicideEngine(game_map, army_1, army_2)
    #CustomPygameView: module vẽ hình ảnh
    view = CustomPygameView(game_map, engine.armies) 
    view.set_nature_units(tree_units) #Thêm cây cối vào để vẽ

    # 5. Xử lý max_turns
    if max_turns == -1:
        run_turns = sys.maxsize
    else:
        run_turns = max_turns

    print("\n>>> BẮT ĐẦU TRẬN ĐẤU...")
    print(">>> Phím SPACE: Pause | S: Step | +/-: Speed")

    try:
        engine.run_game(max_turns=run_turns, view=view, logic_speed=2) #Bật chế độ đồ hoạ view = view, logic_speed = 2: tốc độ vừa phải kịp mắt người nhìn
    except KeyboardInterrupt:
        print("\n>>> Dừng trận đấu.")

    # 6. Báo cáo (Truyền thống kê ban đầu vào) #Sau khi vòng lặp game kết thúc (thắng/thua/hoà/tắt game) =>> gọi hàm tạo báo cáo, truyền vào số liệu ban đầu và kết quả cuối cùng
    # kết quả cuối cùng: engine.winner
    generate_battle_report(engine, engine.winner, init_stats_1, init_stats_2, army_1, army_2)


if __name__ == "__main__":

    run_gui_match()


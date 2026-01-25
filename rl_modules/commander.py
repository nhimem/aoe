import random
import math
from ai.general import General
from core.unit import Unit, UC_BUILDING
# Import đúng kiến trúc cũ
from extensions.custom_units import GameCastle, House

# Các chiến thuật cấp cao
STRATEGY_ATTACK_BASE = 0  # Ưu tiên phá công trình (Castle > House)
STRATEGY_HUNT_UNITS = 1  # Săn quân địch
STRATEGY_MIXED = 2  # Hỗn hợp


class RLCommander(General):
    def __init__(self, army_id: int, role_config: str = "team1", learning=True):
        super().__init__(army_id) #id team, AI đang điều khiển quân nào
        self.role_config = role_config #chiến thuật của đội

        # State: (Tỉ lệ máu, Castle Status, House Status): 1 trạng thái !
        self.q_table = {} #lưu giá trị Q(state,action). Ở State nào thì nên chọn Action nào?
        self.learning_rate = 0.1 #Tốc độ học. chỉ học 10% thông tin mới mỗi lần
        self.discount_factor = 0.95 #Tầm quan trọng của phần thưởng trong tương lai
        self.epsilon = 0.3 if learning else 0.0 #Hệ số tham lam, 30% AI sẽ khám phá chiến thuật mới, thay vì chọn cái tốt hiện tại

        self.last_state = None
        self.last_action = None
        self.previous_score = 0

    def calculate_weighted_score(self, my_units, enemy_units): #Hàm tính điểm
        """
        my_units, enemy_units: list
        => chứa các unite: lính, nhà cửa còn sống trên bản đồ.
        Tính điểm thưởng step-by-step:
        - GameCastle: 50 điểm (Rất quan trọng)
        - House: 10 điểm (Quan trọng)
        - Lính: 1 điểm
        """
        score = 0
        for u in my_units:
            score += u.current_hp #cộng điểm theo máu quân mình

        for u in enemy_units:
            if isinstance(u, GameCastle):
                score -= u.current_hp * 50 #Địch còn castle => trừ điểm nặng
            elif isinstance(u, House):
                score -= u.current_hp * 10 #Địch còn house => trừ điểm vừa
            else:
                score -= u.current_hp #Địch còn lính, trừ điểm nhẹ
        return score
        #Ở enemy_units => điểm bị trừ => mục tiêu là làm máu địch về 0, máu địch giảm => score tăng => điều AI muốn
        """
        Reward Shaping:  thiết kế hàm phần thưởng để hướng dẫn AI học nhanh hơn. 
                + Nếu chỉ thưởng khi thắng game => AI học lâu
                + thưởng theo máu => AI nhận phản hồi lieen tục.
        """
        #Chuyển đổi tình thế trận đấu phức tạp thành 1 định dạng đơn giản để lưu vào Q-table
        #Trừu tượng hoá trạng thái
    def _get_state_key(self, my_units, enemy_units):
        #Tính tổng HP của từng phe
        my_hp = sum(u.current_hp for u in my_units)
        en_hp = sum(u.current_hp for u in enemy_units)

        # 1. Tỷ lệ lực lượng
        ratio = my_hp / (en_hp + 1) # +1: tránh lỗi chia cho 0 khi địch chết hết
        if ratio > 1.2:
            r_state = 2 # team mình mạnh hơn 20%
        elif ratio < 0.8:
            r_state = 0 # team mình yêu hơn 20%
        else:
            r_state = 1 #hai bên cân bằng
        #Gom hàng nghìn trường hợp tương tự nhau vào làm 1 trong 3 trạng thái ở trên !
        #Giống automate, các chuỗi có tính chất tương tự nhau thì cùng ở 1 trạng thái

        # 2. Check trạng thái công trình: mục tiêu quan trọng
        #AI cần biết mục tiên ưu tiên còn tồn tại hay không
        castle_alive = 0 #Chiến thuật phá nhà chính => vô nghĩa
        house_alive = 0

        for u in enemy_units:
            if isinstance(u, GameCastle): #lọc đối tượng trong danh sách lính địch
                castle_alive = 1
            if isinstance(u, House):
                house_alive = 1
            if castle_alive and house_alive: break #break => dừng vòng lặp ngay khi cả 2 công trình vẫn còn

        return (r_state, castle_alive, house_alive) #Trả về tuple => tuple có thể làm key cho dictionnary (q_table)


        #Các trạng thái có thể xảy ra : 3.2.2 = 12 trạng thái khác nhau mà AI có thể gặp
        #Vì sao không đưa toạ độ x,y vào: Số lượng trạng thái sẽ bùng nổ => Ai sẽ phải học lại từ đầu chỉ vì 1 con lính nhích sang phải 1 bước
        #Trạng thái càng đơn giản, AI học càng nhanh.

    def learn_terminal_result(self, final_reward):
        #Rút kinh nghiệm sau khi trận đấu hoàn toàn kết thúc
        #AI nhìn lại hành động cuối cùng mình đã làm và xem nó đáng giá bao nhiêu điểm
        """
        Q table là 1 dict
        key: state_key: trạng thái (tỉ lệ máu, castle_alive, house_alive)
        value: 1 danh sách gồm 3 float, đại diện cho 3 chiến thuật khai báo ở đầu code:
            điểm phá nhà, điểm săn lính, điểm hỗn hợp.
            Ví dụ: (1,1,1) : (15.2, 5.0, 8.4)
            => Chiến thuật Phá nhà(15.2) có điểm cao nhất => Khả năng mang lại chiến thắng cao => Chọn
        """
        #Kiểm tra xem trận đấu có thực sự diễn ra không
        if self.last_state is not None and self.last_action is not None:

            #Tìm trong từ điển q_table khoá tên là self.last_state, nếu tìm được thì trả về 1 List
            #Nếu không tìm thấy, trả về List mặc định [0.0, 0.0, 0.0], nhưng không thay đổi q-table, không thêm key: last_state vào q-table (lần đầu gặp trạng thái này)
            #[self.last_action]: Chọn dòng ghi điểm của hành động mà AI vừa thực hiện (trong List)
            #last_action là 0,1 hoặc 2, phụ thuộc hành động cuối AI chọn.
            #old_q: con số ước tính cũ trong bộ nhớ
            old_q = self.q_table.get(self.last_state, [0.0] * 3)[self.last_action]

            #Công thức cập nhật Q cho trạng thái kết thúc (Terminal State)
            #Điều chỉnh giá trị cũ dựa trên sai số so với thực tế
            #(final_reward - old_q): Temporal Difference Error: khoảng cách giữa kết quả thực tế và dự đoán cũ
            #new_q: là 1 con số mới hợp lý hơn cũ.
            #learning_rate: Nó cần nhiều trận thắng để khẳng định 1 hành động là tốt
            new_q = old_q + self.learning_rate * (final_reward - old_q)

            #Ghi đè giá trị mới vào bộ não Q-table, nếu chưa có last_state thì thêm vào List [0.0]*3 đã
            #Sau đó truy cập vào index last_action trong List để ghi gias trị new_q vào.
            if self.last_state not in self.q_table: self.q_table[self.last_state] = [0.0] * 3
            self.q_table[self.last_state][self.last_action] = new_q


    #được gọi liên tục mỗi frame của trận đấu
    #Vừa đánh, vừa học, vừa ra lệnh
    def decide_actions(self, current_map, my_units, enemy_units):

        #Quan sát và tính toán phần thưởng:
        actions = []
        state_key = self._get_state_key(my_units, enemy_units) #Nhìn chiến trường để lấy state hiện tại.

        # Tính Reward
        #Điểm số hiện tại
        current_score = self.calculate_weighted_score(my_units, enemy_units)
        #Chênh lệch điểm số: reward > 0 => vừa giết được lính địch, điểm tăng
        reward = (current_score - self.previous_score) - 1  # Phạt thời gian:
        # -1: Giúp AI không lười biếng. Cứ mỗi frame trôi qua, AI bị từ 1 điểm.
        #=> Ép buộc AI phải tấn công nhanh để thắng, thay vì đứng yên hưởng thụ số điểm hiện có.
        self.previous_score = current_score

        #Cập nhật trí nhớ: thực hiện việc học ngay trong trận đấu.
        # Q-Learning Update
        if self.last_state is not None and self.last_action is not None:
            old_q = self.q_table.get(self.last_state, [0.0] * 3)[self.last_action]

            #Ước tính ở trạng thái hiện tại, hành động tốt nhất sẽ đem lại bao nhiêu điểm
            max_future_q = max(self.q_table.get(state_key, [0.0] * 3))

            #Công thức Bellman đầy đủ: có thêm self.discount_factor * max_future_q
            #=> AI không chỉ nhìn vào phần thưởng vừa nhận (reward) mà còn nhìn về phía trước
            #=> Nếu làm hành động này, sẽ dẫn tới 1 trạng thái mới có tiềm năng đạt bao nhiêu điểm
            new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)

            #lưu lại vào Q-table.
            if self.last_state not in self.q_table: self.q_table[self.last_state] = [0.0] * 3
            self.q_table[self.last_state][self.last_action] = new_q

        # Chọn Action: Chọn chiến thuật: AI sẽ chơi theo kiểu Phá nhà, Săn lính hay Hỗn hợp.
        if random.random() < self.epsilon:
            strategy = random.randint(0, 2) #thử nghiệm ngẫu nhiên để chọn exploration hay exploitaion
        else:
            qs = self.q_table.get(state_key, [0.0] * 3)
            strategy = qs.index(max(qs)) #Exploitation hành động có điểm cao nhất

        self.last_state = state_key #Lưu lại để frame sau còn học
        self.last_action = strategy

        # --- TÌM MỤC TIÊU --- Xác định mục tiêu
        target_castle = None
        target_house = None

        for u in enemy_units:
            if isinstance(u, GameCastle): #Ưu tiên tìm castle trước
                target_castle = u
                break
            elif isinstance(u, House) and target_house is None:
                target_house = u

        # Ưu tiên: Castle > House : Có castle thì đánh Castle, không thì đánh House.
        target_building = target_castle if target_castle else target_house

        # Danh sách lính địch (không phải nhà): Lọc ra danh sách lích địch để săn
        enemy_troops = [u for u in enemy_units if UC_BUILDING not in u.armor_classes]

        # --- THỰC HIỆN ---: Ra lệnh cụ thể cho từng unit
        #AI duyệt qua từng con lính để giao việc
        for unit in my_units:
            # Lính chết rồi hoặc công trình (nhà) thì không nhận lệnh được dù nằm trong my_units
            if not unit.is_alive or UC_BUILDING in unit.armor_classes: continue

            #Lấy tên loại quân: Knight, Pikeman,..
            unit_type = unit.__class__.__name__
            target = None

            #Dựa vào strategy đã chọn ở khối 3, mỗi quân lính tìm cho mình 1 đích để hướng tới.
            # Chiến thuật 0: Phá nhà => ưu tiên target_builing => Nếu không còn nhà nào, tìm quân địch gần nhất để đánh
            if strategy == STRATEGY_ATTACK_BASE:
                target = target_building
                if not target: target = self.find_closest_enemy(unit, enemy_troops)

            # Chiến thuật 1: Săn lính => Ưu tiên tìm quân địch gần nhất, nếu địch lính hết => phá nhà
            elif strategy == STRATEGY_HUNT_UNITS:
                target = self.find_closest_enemy(unit, enemy_troops)
                if not target: target = target_building

            # Chiến thuật 2: Hỗn hợp: thể hiện sự khác biệt phe (role_config)
            # Nếu là team1 => Crossbowman và Pikeman đi phá nhà. Các quân khác đi săn lính
            # Nếu là team2 => Crossbowman và Knight đi phá nhà
            # Knight chạy nhanh nên phá nhà hpawjc bắt nỏ đinh bịch => hiệu quả hơn

            elif strategy == STRATEGY_MIXED:
                if self.role_config == "team1":
                    if unit_type in ["Crossbowman", "Pikeman"]:
                        target = target_building
                    else:
                        target = self.find_closest_enemy(unit, enemy_troops)
                else:
                    if unit_type in ["Crossbowman", "Knight"]:
                        target = target_building
                    else:
                        target = self.find_closest_enemy(unit, enemy_troops)

                #Sau khi đi qua các bộ lọc "team1/team2" => đơn vị quân đó đã tìm được mục tiêu chưa
                #VD: STRATEGY_MIXED - lính: knight => ưu tiên săn lính địch (find_closest_enemy)
                #=> Nếu quân địch hết lính => target vẫn là non
                #=> Dòng này để phát hiện nếu vẫn là None => giao việc mới, không cho lính đứng chơi
                if not target: target = target_building if target_building else self.find_closest_enemy(unit,
                                                                                                        enemy_troops)

            # Tạo lệnh di chuyển/tấn công
            if target: #Kểm tra xem biến target có tồn tại không
                dist = unit._calculate_distance(target) #Tính khoảng cách từ vị trí hiện tại của quân ta đến mục tiêu
                if dist > unit.attack_range + 0.5: #So sánh khoảng cách với tầm bắn của lính
                    
                    # Di chuyển phân tán nhẹ => Chống chồng lấn, nếu 20 con lính cùng đi đến toạ độ mục tiêu => kẹt
                    # rx,ry giúp lính đứng tản ra xung quanh tự nhiên
                    rx = random.uniform(-1.0, 1.0) #tạo ra độ lệch ngẫu nhiên trong khoảng -1 -> 1
                    ry = random.uniform(-1.0, 1.0)
                    move_pos = (target.pos[0] + rx, target.pos[1] + ry) #tính toán toạ độ đích đến cuối cùng 
                    # toạ độ đích = toạ độ mục tiêu + độ lệch vừa tính
                    actions.append(("move", unit.unit_id, move_pos)) #thêm lệnh mov vào danh sách gửi về GameEngine
                    # "tên lệnh", id_quân_ta, toạ độ đến
                else:
                    actions.append(("attack", unit.unit_id, target.unit_id))
                    # dist nhỏ hơn hoặc bằng tầm bắn => đủ gần => AI chuyển sang lệnh tấn công, Engine tự xử lý việc trừ máu

            # Hit & Run cho Cung thủ
            if unit_type == "Crossbowman": #chỉ có crossbowman mới có hit and run
                closest = self.find_closest_enemy(unit, enemy_troops) #tìm địch gần nhất 
                if closest and unit._calculate_distance(closest) < 2.5: #nếu địch cách crossbowman hơn 2.5 => vẫn đứng bắn bình thường, nếu lọt vào vòng 2.5 => hit and run
                    dx = unit.pos[0] - closest.pos[0] #Toạ độ của mình - Toạ độ địch => Vector hướng về phía mình => Hướng chạy trốn
                    dy = unit.pos[1] - closest.pos[1]
                    mag = math.sqrt(dx * dx + dy * dy) #Khoảng cách thực tế giữa quân ta và địch
                    if mag > 0:
                        run_pos = (unit.pos[0] + dx / mag * 3, unit.pos[1] + dy / mag * 3) #Chuẩn hoá vector, dx/mag => tạo ra vector đơn vị (có độ dài = 1)
                        # nhân với 3 để ép lính chạy 1 quảng đường đúng bằng 3 đơn vị. => Ổn định trong di chuyển
                        actions.append(("move", unit.unit_id, run_pos)) #Ghi đè lệnh lên danh sách actions.
                        #Đoạn code này nằm sau đoạn chọn mục tiêu tấn công => khi append 1 lệnh mov cho cùng 1 unit_id vào list, nếu Engine được lập trình để ưu tiên lệnh cuối cùng
                        #lệnh "move" sẽ huỷ bỏ lệnh "attack" ở trên => crossbowman đang định bắn, thấy địch quá gần => bỏ ý định bắn và chạy

        return actions
    
    """Lập trình game:
    - Phần não (AI): chỉ đưa ra quyết định 
    - Phần xác (Engine): phần mềm chạy ngầm, nhận list actions[] và làm việc nặng nhọc như: vẽ lính di chuyển trên màn hình, tính toán va chạm, trừ máu,..
    """





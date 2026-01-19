import random
import math
from ai.general import General
from core.unit import Unit, UC_BUILDING
from extensions.custom_units import GameCastle, House

# Các hành động cấp cao
STRATEGY_ATTACK_CASTLE = 0  # Ưu tiên tuyệt đối phá nhà
STRATEGY_HUNT_UNITS = 1  # Săn quân địch
STRATEGY_MIXED = 2  # Phân chia role


class RLCommander(General):
    def __init__(self, army_id: int, role_config: str = "team1", learning=True):
        super().__init__(army_id)
        self.role_config = role_config

        # State: (Tỉ lệ máu, Castle địch còn sống hay không)
        # 0: Thua thiệt, 1: Cân bằng, 2: Ưu thế
        # 0: Castle sập, 1: Castle còn
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95  # Tăng tầm nhìn xa
        self.epsilon = 0.3 if learning else 0.0  # Tăng tỉ lệ khám phá

        self.last_state = None
        self.last_action = None
        self.previous_score = 0

    def calculate_weighted_score(self, my_units, enemy_units):
        """Tính điểm: Ưu tiên cực lớn cho việc phá nhà"""
        score = 0
        # Điểm quân ta (1 HP = 1 điểm)
        for u in my_units:
            score += u.current_hp

        # Điểm quân địch (trừ điểm)
        for u in enemy_units:
            if isinstance(u, (GameCastle, House)):
                # QUAN TRỌNG: Máu nhà tính hệ số x10
                # Phá 10 máu nhà = Lời 100 điểm.
                # Bị lính chém mất 50 máu = Lỗ 50 điểm.
                # -> Tổng vẫn Lời 50 -> AI sẽ chọn đánh nhà bất chấp bị đánh.
                score -= u.current_hp * 10
            else:
                score -= u.current_hp
        return score

    def _get_state_key(self, my_units, enemy_units):
        my_hp = sum(u.current_hp for u in my_units)
        en_hp = sum(u.current_hp for u in enemy_units)

        # 1. Tỷ lệ lực lượng
        ratio = my_hp / (en_hp + 1)
        if ratio > 1.2:
            r_state = 2
        elif ratio < 0.8:
            r_state = 0
        else:
            r_state = 1

        # 2. Castle Status (Quan trọng)
        castle_alive = 0
        for u in enemy_units:
            if isinstance(u, GameCastle):
                castle_alive = 1
                break

        return (r_state, castle_alive)

    def learn_terminal_result(self, final_reward):
        if self.last_state is not None and self.last_action is not None:
            old_q = self.q_table.get(self.last_state, [0.0] * 3)[self.last_action]
            new_q = old_q + self.learning_rate * (final_reward - old_q)
            if self.last_state not in self.q_table: self.q_table[self.last_state] = [0.0] * 3
            self.q_table[self.last_state][self.last_action] = new_q

    def decide_actions(self, current_map, my_units, enemy_units):
        actions = []
        state_key = self._get_state_key(my_units, enemy_units)

        # Tính Reward dựa trên Weighted Score
        current_score = self.calculate_weighted_score(my_units, enemy_units)
        # Thêm phạt theo thời gian (-1 mỗi turn) để ép đánh nhanh
        reward = (current_score - self.previous_score) - 1
        self.previous_score = current_score

        # Q-Learning Update
        if self.last_state is not None and self.last_action is not None:
            old_q = self.q_table.get(self.last_state, [0.0] * 3)[self.last_action]
            max_future_q = max(self.q_table.get(state_key, [0.0] * 3))
            new_q = old_q + self.learning_rate * (reward + self.discount_factor * max_future_q - old_q)

            if self.last_state not in self.q_table: self.q_table[self.last_state] = [0.0] * 3
            self.q_table[self.last_state][self.last_action] = new_q

        # Action Selection
        if random.random() < self.epsilon:
            strategy = random.randint(0, 2)
        else:
            qs = self.q_table.get(state_key, [0.0] * 3)
            strategy = qs.index(max(qs))

        self.last_state = state_key
        self.last_action = strategy

        # --- EXECUTION ---
        enemy_castle = None
        for u in enemy_units:
            if isinstance(u, GameCastle):
                enemy_castle = u
                break
            if isinstance(u, House) and enemy_castle is None:
                enemy_castle = u

        target_building = enemy_castle if enemy_castle else (enemy_units[0] if enemy_units else None)

        for unit in my_units:
            if not unit.is_alive or UC_BUILDING in unit.armor_classes: continue

            unit_type = unit.__class__.__name__
            target = None

            # Logic chọn mục tiêu
            if strategy == STRATEGY_ATTACK_CASTLE:
                target = target_building

            elif strategy == STRATEGY_HUNT_UNITS:
                # Săn lính, nếu hết lính thì đánh nhà
                target = self.find_closest_enemy(unit, [u for u in enemy_units if UC_BUILDING not in u.armor_classes])
                if not target: target = target_building

            elif strategy == STRATEGY_MIXED:
                # Logic Team 1 / Team 2
                if self.role_config == "team1":
                    if unit_type in ["Crossbowman", "Pikeman"]:
                        target = target_building
                    else:
                        target = self.find_closest_enemy(unit,
                                                         [u for u in enemy_units if UC_BUILDING not in u.armor_classes])
                else:
                    if unit_type in ["Crossbowman", "Knight"]:
                        target = target_building
                    else:
                        target = self.find_closest_enemy(unit,
                                                         [u for u in enemy_units if UC_BUILDING not in u.armor_classes])

                # Fallback
                if not target: target = target_building

            # Tạo lệnh
            if target:
                dist = unit._calculate_distance(target)
                if dist > unit.attack_range + 0.5:
                    # Micro-move: Thêm nhiễu ngẫu nhiên nhỏ vào đích đến để lính không chồng đống
                    # Giúp lính bao vây castle thay vì xếp hàng 1
                    rx = random.uniform(-1.5, 1.5)
                    ry = random.uniform(-1.5, 1.5)
                    move_pos = (target.pos[0] + rx, target.pos[1] + ry)
                    actions.append(("move", unit.unit_id, move_pos))
                else:
                    actions.append(("attack", unit.unit_id, target.unit_id))

            # Logic Hit & Run đơn giản cho cung thủ
            if self.role_config == "team1" and unit_type == "Crossbowman":
                closest = self.find_closest_enemy(unit, [u for u in enemy_units if UC_BUILDING not in u.armor_classes])
                if closest and unit._calculate_distance(closest) < 2.5:
                    # Kéo dãn khoảng cách
                    dx = unit.pos[0] - closest.pos[0]
                    dy = unit.pos[1] - closest.pos[1]
                    mag = math.sqrt(dx * dx + dy * dy)
                    if mag > 0:
                        run_pos = (unit.pos[0] + dx / mag * 4, unit.pos[1] + dy / mag * 4)
                        actions.append(("move", unit.unit_id, run_pos))

        return actions
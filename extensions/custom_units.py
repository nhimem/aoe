from core.unit import Unit, Wonder, UC_BUILDING

# Giữ nguyên GameCastle và House
class GameCastle(Wonder):
    def __init__(self, unit_id: int, army_id: int, pos: tuple[float, float]):
        super().__init__(unit_id, army_id, pos)
        self.max_hp = 1000
        self.current_hp = 1000
        self.hitbox_radius = 2.5

class House(Unit):
    def __init__(self, unit_id: int, army_id: int, pos: tuple[float, float]):
        super().__init__(
            unit_id=unit_id, army_id=army_id, pos=pos,
            hp=250, speed=0.0, attack_power=0, attack_range=0,
            attack_type="melee", melee_armor=5, pierce_armor=5, line_of_sight=4,
            armor_classes=[UC_BUILDING, "Standard Buildings"],
            bonus_damage={},
            hitbox_radius=1.2,
            reload_time=999.0
        )

class NatureTree(Unit):
    """
    Cây cối: Lưu thêm thông tin để View biết vẽ ảnh nào.
    """
    def __init__(self, unit_id: int, army_id: int, pos: tuple[float, float], tree_type: int = 1, variant: int = 0):
        super().__init__(
            unit_id=unit_id, army_id=army_id, pos=pos,
            hp=10000, speed=0.0, attack_power=0, attack_range=0,
            attack_type="melee", melee_armor=100, pierce_armor=100, line_of_sight=0,
            armor_classes=["Nature"],
            bonus_damage={},
            hitbox_radius=0.5, # Tương ứng 1 ô vuông (đường kính 1.0)
            reload_time=999.0
        )
        self.is_alive = True
        # tree_type: 1, 2 (Team 1) hoặc 3, 4 (Team 2)
        # variant: 0..6 (Index của ảnh trong folder)
        self.tree_type = tree_type
        self.variant = variant
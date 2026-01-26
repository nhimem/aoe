from core.unit import Unit, Wonder, UC_BUILDING
#Unit: Class cơ sở cho mọi thực thể trong game (lính, nhà, cây). Chứa các chỉ số như: máu, tốc độ, giáp
#Wonder: unit đặc biệt => nhà chính
#UC_BUILDING: đánh dấu 1 đơn vị thuộc lớp công trình

#Tạo ra 3 class đại diện cho Castle (Lâu đài), House (nhà dân), và Cây cối (NatureTree)
# Giữ nguyên GameCastle và House
class GameCastle(Wonder): #Mất GameCastle => thua game.

    #unit_id: ID duy nhất của unit
    #army_id: ID của đội
    #pos: toạ độ x,y
    
    def __init__(self, unit_id: int, army_id: int, pos: tuple[float, float]):
        super().__init__(unit_id, army_id, pos)
        self.max_hp = 1000 #thiết lập HP tối đa
        self.current_hp = 1000 #HP hiện tại
        self.hitbox_radius = 2.5 #Bán kính va chạm, chiếm vòng tròn đường kính 5.0 trên bản đồ, các unite khác không thể đi xuyên qua vùng này

class House(Unit): #Công trình phụ, chặn đường hoặc làm mục tiêu giả
    def __init__(self, unit_id: int, army_id: int, pos: tuple[float, float]):
        super().__init__(
            unit_id=unit_id, army_id=army_id, pos=pos,
            hp=250, speed=0.0, attack_power=0, attack_range=0, #speed = 0 => đứng yên, không di chuyển, không có sát thương
            attack_type="melee", melee_armor=5, pierce_armor=5, line_of_sight=4, #giáp vật lý và giáp xuyên thấu là 5, tầm nhìn là 4 ô
            armor_classes=[UC_BUILDING, "Standard Buildings"], #gán nhãn cho unit này là UC_BUILDING.
            bonus_damage={}, #Nếu lính địch có chỉ số bonus_damage chống lại Building, House sẽ nhận thêm sát thương
            hitbox_radius=1.2, #Kích thước chiếm chỗ trong map.
            reload_time=999.0 #Thời gian nạp đạn cực lâu => Vô hiệu hoá vì không bao giờ đánh.
        )
 
class NatureTree(Unit): #Vật cản môi trường.
    """
    Cây cối: Lưu thêm thông tin để View biết vẽ ảnh nào.
    """
    def __init__(self, unit_id: int, army_id: int, pos: tuple[float, float], tree_type: int = 1 : #loại cây, variant: int = 0: #biến thể):
        super().__init__(
            unit_id=unit_id, army_id=army_id, pos=pos,
            hp=10000, speed=0.0, attack_power=0, attack_range=0, #máu cực lớn, giám cực to, không có tầm nhìn, vật vô tri.
            attack_type="melee", melee_armor=100, pierce_armor=100, line_of_sight=0,
            armor_classes=["Nature"], #gán nhãn Nature cho class
            bonus_damage={}, 
            hitbox_radius=0.5, # Tương ứng 1 ô vuông (đường kính 1.0) #tương ứng 1 ô vuông.
            reload_time=999.0
        )
        self.is_alive = True
        # tree_type: 1, 2 (Team 1) hoặc 3, 4 (Team 2)
        # variant: 0..6 (Index của ảnh trong folder)

        
        self.tree_type = tree_type 
        self.variant = variant
        #lưu trữ lại 2 tham số đã truyền vào để sau này lớp CustopPygameView (biên file view) đọc ra và biết chọn file ảnh .png nào để vẽ lên màn hình


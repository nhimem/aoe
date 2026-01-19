import random
from core.map import Map
from core.unit import Knight, Pikeman, Crossbowman
from extensions.custom_units import GameCastle, House, NatureTree


def create_battle_map(width=80, height=80):
    game_map = Map(width, height)
    tree_list = []  # Danh sách chứa object cây để View vẽ

    # Số lượng cây
    num_trees = 300

    for i in range(num_trees):
        tx = random.randint(0, width - 1)
        ty = random.randint(0, height - 1)

        # Tránh spawn đè lên vùng căn cứ 2 đội (Góc 15,15 và 105,105 hoặc tùy size map)
        margin = 10
        if (tx < margin and ty < margin) or (tx > width - margin and ty > height - margin):
            continue

        # Logic chọn loại cây (Visual)
        if tx < width / 2:
            tree_type = random.choice([1, 2])
        else:
            tree_type = random.choice([3, 4])

        variant = random.randint(0, 6)

        # Tạo Unit Cây (Lưu ý: army_id=None để biết nó không thuộc phe nào)
        tree = NatureTree(unit_id=900000 + i, army_id=None, pos=(float(tx), float(ty)), tree_type=tree_type,
                          variant=variant)
        tree_list.append(tree)

        # QUAN TRỌNG: Đánh dấu lên Map đây là vật cản
        # Engine tìm đường sẽ dựa vào cái này
        if 0 <= tx < width and 0 <= ty < height:
            game_map.add_obstacle("tree", tx, ty)

    return game_map, tree_list


def generate_army_composition(army_id, start_x, start_y, total_units=50):
    # (Giữ nguyên code cũ của bạn)
    units = []
    base_id = army_id * 10000

    castle = GameCastle(base_id + 1, army_id, (start_x, start_y))
    units.append(castle)

    house_offsets = [(5, 5), (-5, -5), (5, -5), (-5, 5)]
    for idx, (ox, oy) in enumerate(house_offsets):
        hx, hy = start_x + ox, start_y + oy
        house = House(base_id + 2 + idx, army_id, (hx, hy))
        units.append(house)

    unit_types = [Knight, Pikeman, Crossbowman]
    for i in range(total_units):
        u_class = random.choice(unit_types)
        ux = start_x + random.uniform(-10, 10)
        uy = start_y + random.uniform(-10, 10)
        # Chỉnh lại tọa độ một chút để không kẹt vào nhà
        unit = u_class(base_id + 100 + i, army_id, (ux, uy))
        units.append(unit)

    return units
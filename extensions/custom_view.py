import pygame
import os
from view.gui_view import PygameView, BG_COLOR, RED, GREEN, BLUE, BLACK
from extensions.custom_units import NatureTree, House, GameCastle
#Thêm khả năng vẽ hình ảnh tĩnh + xử lý việc hiển thị 2.5D (vật ở gần che vật ở xa)

#os.path.join => nối chuỗi đường dẫn, tự chọn dấu phân cách phù hợp tuỳ vào OS

# Định nghĩa đường dẫn
ASSET_DIR = "assets/resources" #định nghĩa đường dẫn
BUILDING_DIR = os.path.join(ASSET_DIR, "buildings") #Thư mục chứa ảnh nhà
NATURE_DIR = os.path.join(ASSET_DIR, "natures") #Thư mục chứa ảnh cây


class CustomPygameView(PygameView):
    def __init__(self, map_instance, armies):
        super().__init__(map_instance, armies) #Gọi hàm khởi tạo của cha
        self.nature_units = []  # Danh sách chứa cây cối (vẽ riêng): list riêng chứa cây vì cây không thuộc team nào
        self._load_custom_sprites() #Tải ảnh ngay khi khởi tạo

    def set_nature_units(self, nature_list):
        """
        Nhận danh sách cây từ MapBuilder để vẽ.
        Cây không nằm trong Army để tránh bị tính vào thống kê chiến đấu.
        """
        self.nature_units = nature_list

    def _load_custom_sprites(self): #Tải công trình
        try: 
            # 1. Load Buildings (Castle & House)
            self.custom_images = {
                'castle': self._safe_load_image(os.path.join(BUILDING_DIR, "castle.png")),
                'house1': self._safe_load_image(os.path.join(BUILDING_DIR, "house1.png")), #Nhà team xanh
                'house2': self._safe_load_image(os.path.join(BUILDING_DIR, "house2.png")), #Nhà team đỏ
            } #Tạo từ điển self.custom_images để lưu ảnh công trình

            
            # 2. Load Trees (Nature) Tải cây
            self.tree_images = {}
            # Scale ảnh cây về khoảng 70x80px cho vừa ô grid nhưng vẫn cao ráo
            TARGET_SIZE = (70, 80) #Kích thước chuẩn hoá cho cây => Ảnh gốc của cây có thể rất to hoặc nhỏ. Dòng này ép kích thước về 1 chuẩn để khi vẽ lên bản đồ => không bị lệch grid
            

            for t_type in [1, 2, 3, 4]: #Duyệt qua 4 loại cây
                #VD Cấu trúc lưu trữ là: tree_images[1][0]
                self.tree_images[t_type] = {}
                folder_path = os.path.join(NATURE_DIR, f"tree_{t_type}")

                if not os.path.exists(folder_path): #Kiểm tra thư mục tồn tại
                    continue

                # Lấy danh sách ảnh png, sort để đảm bảo thứ tự variant
                files = sorted([f for f in os.listdir(folder_path) if f.endswith(".png")])
                for idx, filename in enumerate(files):
                    if idx > 6: break  # Chỉ lấy tối đa 7 biến thể

                    img_path = os.path.join(folder_path, filename)
                    raw_img = pygame.image.load(img_path).convert_alpha()
                    scaled_img = pygame.transform.scale(raw_img, TARGET_SIZE)
                    self.tree_images[t_type][idx] = scaled_img

            print(">>> Custom Sprites (Buildings & Trees) Loaded.")

        except Exception as e:
            print(f"Lỗi tải Custom Sprites: {e}")

    def _safe_load_image(self, path):
        """Hàm hỗ trợ tải ảnh an toàn, trả về None nếu lỗi/không thấy file"""
        try:
            if os.path.exists(path):
                return pygame.image.load(path).convert_alpha()
        except Exception:
            pass
        return None

    def get_unit_image(self, unit):
        """
        Trả về ảnh tĩnh cho các Unit tùy chỉnh.
        Trả về None nếu là lính thường (để dùng logic sprite của cha).
        """
        # 1. Cây (NatureTree)
        #Lấy ảnh từ dict tree_images dựa trên tree_type và variant
        if isinstance(unit, NatureTree):
            t_type = getattr(unit, 'tree_type', 1)
            variant = getattr(unit, 'variant', 0)
            return self.tree_images.get(t_type, {}).get(variant, None)

        # 2. Nhà (House): chọn nhà xanh (house1) cho team 0, nhà đỏ cho team 1
        if isinstance(unit, House):
            return self.custom_images['house1'] if unit.army_id == 0 else self.custom_images['house2']

        # 3. Lâu đài (Castle) #Trả về ảnh castle
        if isinstance(unit, GameCastle):
            return self.custom_images['castle']

        return None #Nếu là lính thường => trả về None

    #Vẽ lên màn hình: kỹ thuật DEPTH SORTING: Sắp xếp chiều sâu
    def draw_units(self, armies):
        """
        Ghi đè hoàn toàn hàm vẽ units để xử lý hỗn hợp:
        - Custom Units (Ảnh tĩnh): Cây, Nhà, Lâu đài.
        - Standard Units (Spritesheet): Lính, Kỵ sĩ.
        """
        visible_units = []

        # 1. Thêm Cây vào danh sách vẽ (Gán army_id giả = 99)
        for tree in self.nature_units:
            visible_units.append((99, tree))

        # 2. Thêm Lính/Nhà từ các Army thực
        for army in armies:
            for unit in army.units:
                visible_units.append((army.army_id, unit))

        # 3. Sắp xếp theo trục Y (Depth Sorting) để vật ở dưới che vật ở trên
        # Trong góc nhìn từ trên xuống, vật nào đứng thấp hơn trên màn hình (toạ độ Y lớn hơn) thì ở gần mắt người xem hơn => vẽ sau để đè lên vật ở xa
        # Sắp xếp theo: Y -> X -> Unit ID
        visible_units.sort(key=lambda p: (round(p[1].pos[1], 1), round(p[1].pos[0], 1), p[1].unit_id)) #Sắp xếp danh sách vẽ: vẽ vật ở xa trc, gần sau

        for army_id, unit in visible_units:
            # Bỏ qua unit đã chết (trừ khi là lính đang có animation chết)
            if not unit.is_alive and isinstance(unit, (NatureTree, House, GameCastle)):
                continue

            x, y = unit.pos

            # Kiểm tra nằm trong phạm vi bản đồ
            if not (0 <= int(x) < self.map.width and 0 <= int(y) < self.map.height):
                continue

            # Tính toán tọa độ màn hình
            screen_x, screen_y = self.cart_to_iso(x, y)

            # Tính độ cao địa hình (Elevation) để vẽ đúng độ cao
            tile = self.map.grid[int(x)][int(y)]
            height_offset = 0
            if tile.terrain_type != 'water':
                height_offset = int(tile.elevation * 2 * self.zoom)

            # draw_pos_y là tọa độ pixel tại "mặt đất" (chân unit)
            draw_pos_y = screen_y - height_offset

            # --- VẼ UNIT ---
            custom_img = self.get_unit_image(unit)

            # TRƯỜNG HỢP 1: CUSTOM UNIT (Cây, Nhà, Lâu đài)
            if custom_img:
                # Scale ảnh theo zoom
                if self.zoom != 1.0:
                    w = int(custom_img.get_width() * self.zoom)
                    h = int(custom_img.get_height() * self.zoom)
                    img_to_draw = pygame.transform.scale(custom_img, (w, h))
                else:
                    img_to_draw = custom_img

                # Tọa độ vẽ ảnh (top-left của ảnh)
                # Căn giữa chiều ngang (screen_x) và chân ảnh nằm tại mặt đất (draw_pos_y)
                draw_x = screen_x - img_to_draw.get_width() // 2
                draw_y = draw_pos_y - img_to_draw.get_height()

                self.screen.blit(img_to_draw, (draw_x, draw_y))

                # Vẽ thanh máu (Trừ cây)
                # Vị trí: Trên đỉnh đầu ảnh (draw_y) trừ đi khoảng 10px
                if self.show_hp_bars and not isinstance(unit, NatureTree):
                    self._draw_custom_health_bar(unit, screen_x, draw_y - 10)

            # TRƯỜNG HỢP 2: STANDARD UNIT (Lính thường - Dùng SpriteSheet)
            else:
                # Logic lấy frame hoạt hình tương tự lớp cha
                if not unit.is_alive:
                    death_elapsed = getattr(unit, 'death_elapsed', 0)
                    if death_elapsed > 2000: continue

                sprites_for_unit = self.unit_sprites.get(unit.__class__)
                color_key = 'blue' if army_id == 0 else 'red'
                state = getattr(unit, 'statut', 'idle')
                if state == 'statique': state = 'idle'

                current_frame = None
                if sprites_for_unit:
                    frames_orient = sprites_for_unit.get(color_key, {}).get(state)
                    if frames_orient:
                        orient_idx = getattr(unit, '_last_orient', 0)
                        nframes = len(frames_orient[orient_idx]) if frames_orient[orient_idx] else 0
                        if nframes > 0:
                            idx = getattr(unit, 'anim_index', 0) % nframes
                            try:
                                current_frame = frames_orient[orient_idx][idx]
                            except:
                                pass

                if current_frame:
                    surf = current_frame
                    draw_x = screen_x - surf.get_width() // 2
                    draw_y = draw_pos_y - surf.get_height()
                    self.screen.blit(surf, (draw_x, draw_y))

                    # Vẽ thanh máu
                    if self.show_hp_bars and unit.is_alive:
                        # Vị trí: Trên đỉnh đầu sprite (draw_y) trừ đi khoảng 5px
                        self._draw_custom_health_bar(unit, screen_x, draw_y - 5)
                else:
                    # Fallback (Hình tròn) nếu chưa load xong sprite
                    if unit.is_alive:
                        color = BLUE if army_id == 0 else RED
                        pygame.draw.circle(self.screen, color, (screen_x, draw_pos_y - 10), int(10 * self.zoom))
                        if self.show_hp_bars:
                            self._draw_custom_health_bar(unit, screen_x, draw_pos_y - 30)

    def _draw_custom_health_bar(self, unit, x, y):
        """
        Vẽ thanh máu tại tọa độ (x, y).
        (x, y) nên là điểm phía trên đầu unit.
        """
        if unit.max_hp <= 0: return

        hp_ratio = unit.current_hp / unit.max_hp
        # Kích thước thanh máu thay đổi theo zoom
        w = int(24 * self.zoom)
        h = max(2, int(4 * self.zoom))

        # Vẽ nền đỏ (máu mất)
        pygame.draw.rect(self.screen, (200, 50, 50), (x - w // 2, y, w, h))
        # Vẽ phần xanh (máu còn)
        pygame.draw.rect(self.screen, (50, 200, 50), (x - w // 2, y, int(w * hp_ratio), h))

    def display(self, armies, time_elapsed, paused, speed_multiplier=1.0):
        """
        Override hàm display chính.
        Logic vẽ Map và Unit đã được CustomPygameView xử lý (bao gồm Cây).
        Logic vẽ UI (bảng thống kê) sẽ do lớp cha xử lý (chỉ hiện Army chính).
        """

        return super().display(armies, time_elapsed, paused, speed_multiplier)


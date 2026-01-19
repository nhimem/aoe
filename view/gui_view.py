import pygame
import os
import math
from io import BytesIO
from PIL import Image

from core.map import Map
from core.army import Army
# Import de toutes les classes d'unités pour le mappage des sprites 
from core.unit import (
    Knight, Pikeman, Crossbowman, LongSwordsman, LightCavalry, 
    Castle, Wonder, Unit
)

# --- CONSTANTES DE CONFIGURATION VUE ---
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
BG_COLOR = (30, 30, 30)

# Dimensions isométriques de BASE (Ratio 2:1)
BASE_TILE_WIDTH = 64
BASE_TILE_HEIGHT = 32

# Couleurs
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 50, 50)
BLUE = (50, 50, 200)
GREEN = (50, 200, 50)

# Couleurs Terrain
TERRAIN_COLORS = {
    0: (100, 200, 100),  # Herbe basse
    1: (120, 180, 100),  # Herbe
    2: (140, 160, 80),   # Terre
    3: (160, 140, 60),   # Colline
    4: (180, 120, 40),   # Montagne
}
WATER_COLOR = (60, 100, 200)
OBSTACLE_COLOR = (100, 100, 100) # Gris pour les obstacles

MINIMAP_SIZE = 280
MINIMAP_MARGIN = 20

# Taille d'une seule frame dans VOS spritesheets (estimee).
SPRITE_FRAME_WIDTH = 48  
SPRITE_FRAME_HEIGHT = 72

# Echelle d'affichage des unites (2.5 = taille visible)
UNIT_DISPLAY_SCALE = 2.5 

class PygameView:
    def __init__(self, game_map: Map, armies: list = None):
        pygame.init()
        pygame.display.set_caption("MedievAIl - Isometric View")
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.map = game_map
        self.font = pygame.font.SysFont('Arial', 16, bold=True)
        self.ui_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Store armies for selective sprite loading
        self._armies = armies
        
        # --- Dynamic screen dimensions (for resize/fullscreen support) ---
        self.screen_w = SCREEN_WIDTH
        self.screen_h = SCREEN_HEIGHT

        # --- Zoom Settings (optimized - 4 levels) ---
        self.min_zoom = 0.2  # déZoom loin
        self.max_zoom = 2.0  # Zoom proche
        # Discrete zoom levels for caching (reduced for faster loading)
        self.zoom_levels = [0.2, 0.5, 1.0, 2.0]
        self.zoom = self.min_zoom  # Demarrer dezoome au max
        # Cache for pre-scaled sprites at each zoom level
        self._zoom_cache: dict[float, dict] = {}

        # Dimensions actuelles (seront calculées via update_zoom_metrics)
        self.tile_width = BASE_TILE_WIDTH
        self.tile_height = BASE_TILE_HEIGHT
        self.tile_half_w = self.tile_width // 2
        self.tile_half_h = self.tile_height // 2
        
        # Offsets d'affichage
        self.offset_x = 0
        self.offset_y = 0
        self.update_zoom_metrics()

        # --- Caméra ---
        self.scroll_speed = 15
        self.scroll_speed_fast = 45  # Vitesse avec Maj
        
        # Scroll initial au centre (0,0)
        self.scroll_x = 0
        self.scroll_y = 0
        
        # --- Toggles UI (Req 12 PDF: F1-F4) ---
        self.show_army_info = True      # F1: Infos générales
        self.show_hp_bars = True        # F2: Barres de vie
        self.show_minimap = True        # F3/M: Minimap
        self.show_unit_details = True  # F4: Détails unités
        
        # --- Drag souris pour scroll ---
        self.is_dragging = False
        self.drag_start_x = 0
        self.drag_start_y = 0
        self.drag_scroll_start_x = 0
        self.drag_scroll_start_y = 0
        
        # Debug obstacles removed for production

        # --- Assets Originaux (Chargés une fois à taille de base) ---
        self.orig_grass: pygame.Surface | None = None
        self.orig_tree: pygame.Surface | None = None
        self.orig_units: dict = {}

        # --- Sprites Actuels (Redimensionnés selon zoom) ---
        self.unit_sprites: dict = {}
        self.tree_sprite: pygame.Surface | None = None
        self.grass_sprite: pygame.Surface | None = None
        
        # --- Loading with progress ---
        num_unit_types = 5
        num_zoom_levels = len(self.zoom_levels)
        total_steps = num_unit_types + num_zoom_levels + 1
        
        self._show_loading_screen("Initialisation...", 0, total_steps)
        
        self._load_sprites_with_progress(total_steps)
        
        # Pre-cache ALL zoom levels
        current_zoom = self.zoom
        for idx, z in enumerate(self.zoom_levels):
            step = num_unit_types + idx + 1
            self._show_loading_screen(f"Zoom {z}x", step, total_steps)
            self.zoom = z
            self.update_zoom_metrics()
            self._rescale_assets()
        
        # Restore initial zoom
        self.zoom = current_zoom
        self.update_zoom_metrics()
        self._rescale_assets()
        
        self._show_loading_screen("Pret!", total_steps, total_steps)
        pygame.time.wait(200)
        print("Chargement termine.")
        
        # timing pour les animations cote vue (ms)
        self._last_anim_tick = pygame.time.get_ticks()
        # Facteur pour ralentir l'animation cote vue (>1 = plus lent)
        self.anim_time_scale = 0.5
    def _get_minimap_metrics(self):
        """Calcule les métriques pour le dessin et l'interaction avec la minimap."""
        mm_center_x = self.screen_w - MINIMAP_SIZE / 2 - MINIMAP_MARGIN
        mm_center_y = self.screen_h - MINIMAP_SIZE / 2 - MINIMAP_MARGIN
        max_dim = self.map.width + self.map.height
        scale_mm = (MINIMAP_SIZE * 0.8) / max_dim
        # Offset pour centrer verticalement le losange dans le carré
        offset_y_mm = (self.map.width + self.map.height) * scale_mm * 0.25
        return mm_center_x, mm_center_y, scale_mm, offset_y_mm

    def _cart_to_mm(self, cx, cy):
        """Convertit les coordonnées grille en coordonnées écran sur la minimap."""
        mm_cx, mm_cy, scale, offset_y = self._get_minimap_metrics()
        mx = mm_cx + (cx - cy) * scale
        my = mm_cy + (cx + cy) * scale * 0.5 - offset_y
        return int(mx), int(my)

    def _mm_to_cart(self, mx, my):
        """Convertit les coordonnées écran de la minimap en coordonnées grille."""
        mm_cx, mm_cy, scale, offset_y = self._get_minimap_metrics()
        term1 = (mx - mm_cx) / scale
        term2 = (my - mm_cy + offset_y) / (scale * 0.5)
        cx = (term1 + term2) / 2
        cy = (term2 - term1) / 2
        return cx, cy

    def _get_minimap_rect(self):
        """Retourne le rectangle englobant de la minimap pour la détection de collision."""
        return pygame.Rect(
            self.screen_w - MINIMAP_SIZE - MINIMAP_MARGIN,
            self.screen_h - MINIMAP_SIZE - MINIMAP_MARGIN,
            MINIMAP_SIZE,
            MINIMAP_SIZE
        )

    def _center_camera_on_grid(self, gx, gy):
        """Centre la caméra sur une coordonnée spécifique de la carte."""
        iso_x = (gx - gy) * self.tile_half_w
        iso_y = (gx + gy) * self.tile_half_h
        # Positionner scroll_x/y pour que iso_x/y arrive au centre de l'écran
        self.scroll_x = iso_x + self.offset_x - self.screen_w / 2
        self.scroll_y = iso_y + self.offset_y - self.screen_h / 2
        self._clamp_camera()

    def _show_loading_screen(self, step_text: str, current_step: int, total_steps: int):
        """Affiche un écran de chargement avec progression."""
        # Process events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        # Background
        self.screen.fill((15, 18, 25))
        
        # Title
        title_font = pygame.font.SysFont('Arial', 48, bold=True)
        title = title_font.render("MedievAIl", True, (255, 215, 100))
        self.screen.blit(title, (self.screen_w // 2 - title.get_width() // 2, self.screen_h // 2 - 100))
        
        # Subtitle
        sub_font = pygame.font.SysFont('Arial', 18)
        sub = sub_font.render("Chargement en cours...", True, (120, 120, 140))
        self.screen.blit(sub, (self.screen_w // 2 - sub.get_width() // 2, self.screen_h // 2 - 40))
        
        # Progress bar
        bar_w, bar_h = 400, 20
        bar_x = self.screen_w // 2 - bar_w // 2
        bar_y = self.screen_h // 2 + 10
        
        # Background bar
        pygame.draw.rect(self.screen, (30, 30, 40), (bar_x, bar_y, bar_w, bar_h))
        pygame.draw.rect(self.screen, (50, 50, 60), (bar_x, bar_y, bar_w, bar_h), 2)
        
        # Progress fill
        progress = current_step / max(1, total_steps)
        fill_w = int(bar_w * progress)
        if fill_w > 0:
            # Gradient-like effect
            for i in range(fill_w):
                alpha = 0.7 + 0.3 * (i / bar_w)
                color = (int(80 * alpha), int(180 * alpha), int(255 * alpha))
                pygame.draw.line(self.screen, color, (bar_x + i, bar_y + 2), (bar_x + i, bar_y + bar_h - 2))
        
        # Percentage
        pct_text = self.font.render(f"{int(progress * 100)}%", True, (200, 200, 200))
        self.screen.blit(pct_text, (bar_x + bar_w + 15, bar_y + 2))
        
        # Current step
        step_font = pygame.font.SysFont('Arial', 14)
        step = step_font.render(step_text, True, (100, 100, 120))
        self.screen.blit(step, (self.screen_w // 2 - step.get_width() // 2, bar_y + 35))
        
        # Animated dots
        dots = "." * (1 + (pygame.time.get_ticks() // 500) % 3)
        dots_txt = sub_font.render(dots, True, (100, 100, 120))
        self.screen.blit(dots_txt, (self.screen_w // 2 + sub.get_width() // 2, self.screen_h // 2 - 40))
        
        pygame.display.flip()
    
    def _load_sprites_with_progress(self, total_steps: int):
        """Charge les sprites avec affichage de progression."""
        self._load_sprites(total_steps)

    def update_zoom_metrics(self):
        """Recalcule les dimensions des tuiles et l'offset de base selon le zoom."""
        self.tile_width = int(BASE_TILE_WIDTH * self.zoom)
        self.tile_height = int(BASE_TILE_HEIGHT * self.zoom)
        self.tile_half_w = self.tile_width // 2
        self.tile_half_h = self.tile_height // 2

        # Recalcul de l'offset pour centrer la map (0,0 au centre théorique)
        center_map_x = self.map.width / 2
        center_map_y = self.map.height / 2
        iso_center_x = (center_map_x - center_map_y) * self.tile_half_w
        iso_center_y = (center_map_x + center_map_y) * self.tile_half_h
        # Use dynamic screen dimensions
        self.offset_x = self.screen_w / 2 - iso_center_x
        self.offset_y = self.screen_h / 2 - iso_center_y

    def _load_webp_asset(self, path: str, target_size: tuple[int, int], is_spritesheet: bool) -> pygame.Surface | None:
        """
        Charge un fichier WEBP via Pillow, convertit en raw bytes,
        charge dans Pygame, et met à l'échelle vers la taille CIBLE.
        Optimisé pour éviter l'encodage PNG.
        """
        try:
            pil_img = Image.open(path).convert("RGBA")
            
            if is_spritesheet:
                # Découpe la première frame (position 0,0)
                frame_area = (0, 0, SPRITE_FRAME_WIDTH, SPRITE_FRAME_HEIGHT)
                pil_img = pil_img.crop(frame_area)
            
            # Conversion Pillow -> Bytes (Raw RGBA) -> Pygame Surface
            mode = pil_img.mode
            size = pil_img.size
            data = pil_img.tobytes()
            
            surface = pygame.image.frombytes(data, size, mode).convert_alpha()

            # Mise à l'échelle (pour l'affichage isométrique)
            return pygame.transform.scale(surface, target_size)

        except FileNotFoundError:
            print(f"Erreur FATALE: Fichier non trouvé: {path}")
            return None
        except Exception as e:
            print(f"Erreur de chargement de l'asset {path}: {e}")
            return None

    def _load_spritesheet_grid(self, path: str, rows: int, cols: int) -> list[list[pygame.Surface]] | None:
        """Charge une spritesheet WebP et découpe en grille [rows][cols].
        Retourne une liste de listes: frames[row][col].
        Optimisé via tobytes/frombytes.
        """
        try:
            pil_img = Image.open(path).convert("RGBA")
        except FileNotFoundError:
            return None
        except Exception as e:
            print(f"Erreur ouverture spritesheet {path}: {e}")
            return None

        sheet_w, sheet_h = pil_img.size
        # Déduire la taille d'une frame si possible
        frame_w = sheet_w // cols if cols > 0 else 0
        frame_h = sheet_h // rows if rows > 0 else 0
        if frame_w <= 0 or frame_h <= 0:
            return None

        frames: list[list[pygame.Surface]] = []
        for r in range(rows):
            row_frames: list[pygame.Surface] = []
            for c in range(cols):
                left = c * frame_w
                upper = r * frame_h
                right = left + frame_w
                lower = upper + frame_h
                try:
                    frame = pil_img.crop((left, upper, right, lower))
                    
                    mode = frame.mode
                    size = frame.size
                    data = frame.tobytes()
                    surf = pygame.image.frombytes(data, size, mode).convert_alpha()
                    
                    row_frames.append(surf)
                except Exception:
                    # ignorer frame invalide
                    continue
            frames.append(row_frames)
        return frames

    def _load_cache_frames(self, cache_dir: str, target_size: tuple[int,int]) -> list[pygame.Surface] | None:
        """Charge les images individuelles dans `assets/.cache/<name>`.
        Renvoie une liste de Surfaces déjà mises à l'échelle, ou None si aucun fichier.
        """
        try:
            if not os.path.isdir(cache_dir):
                return None
            files = sorted(f for f in os.listdir(cache_dir) if f.lower().endswith(('.png', '.webp', '.jpg', '.jpeg'))) 
            frames: list[pygame.Surface] = []
            for fn in files:
                path = os.path.join(cache_dir, fn)
                try:
                    pil = Image.open(path).convert("RGBA")
                    mode = pil.mode
                    size = pil.size
                    data = pil.tobytes()
                    surf = pygame.image.frombytes(data, size, mode).convert_alpha()
                    
                    surf = pygame.transform.scale(surf, target_size)
                    frames.append(surf)
                except Exception:
                    # Ignorer les fichiers invalides
                    continue
            return frames if frames else None
        except Exception as e:
            print(f"Erreur lors du chargement du cache {cache_dir}: {e}")
            return None

    def _load_sprites(self, total_steps: int = 17):
        """
        Charge tous les assets graphiques nécessaires dans self.orig_*.
        Gère les extensions .webp et .png dynamiquement.
        """
        BASE_PATH = "assets" 
        
        # 1. Chargement des Originaux (Taille Zoom = 1.0)
        # Gazon
        grass_path = os.path.join(BASE_PATH, "resources/grass/grass.webp")
        self.orig_grass = self._load_webp_asset(
            grass_path, 
            target_size=(BASE_TILE_WIDTH, BASE_TILE_WIDTH // 2 + 16), 
            is_spritesheet=False
        ) 

        # Arbre
        tree_path = os.path.join(BASE_PATH, "resources/tree/Tree.webp")
        self.orig_tree = self._load_webp_asset(
            tree_path, 
            target_size=(60, 90), 
            is_spritesheet=False
        ) 
        
        # Mapping des classes vers les dossiers d'assets
        all_unit_configs = {
            Knight: "knight",
            Crossbowman: "crossbowman",
            Pikeman: "pikeman",
            LongSwordsman: "longswordman",
            LightCavalry: "knight", # Fallback sur les sprites de chevalier
        }
        
        # Optimisation : charger uniquement les unités présentes
        needed_unit_classes = set()
        if self._armies:
            for army in self._armies:
                for unit in army.units:
                    needed_unit_classes.add(unit.__class__)
                    if unit.__class__ == LightCavalry:
                        needed_unit_classes.add(Knight)
        
        unit_configs = all_unit_configs if not needed_unit_classes else {
            k: v for k, v in all_unit_configs.items() if k in needed_unit_classes
        }

        states_grid = {
            'death': (24, 30),
            'idle': (24, 30),
            'walk': (16, 30),
            'attack': (16, 30),
        }

        self.orig_units = {}
        for idx, (u_class, name) in enumerate(unit_configs.items()):
            self._show_loading_screen(f"Sprites: {name}", idx + 1, total_steps)
            
            self.orig_units[u_class] = {'blue': {}, 'red': {}}
            for color_folder, prefix in (('blue', 'b'), ('red', 'r')):
                for state, (rows, cols) in states_grid.items():
                    frames = None
                    # Correction : Essayer .webp puis .png pour supporter le Crossbowman
                    for ext in [".webp", ".png"]:
                        filename = f"{prefix}_{name}_{state}{ext}"
                        path = os.path.join(BASE_PATH, 'units', name, color_folder, filename)
                        
                        if os.path.exists(path):
                            frames = self._load_spritesheet_grid(path, rows, cols)
                            if frames:
                                break # Fichier trouvé et chargé
                    
                    self.orig_units[u_class][color_folder][state] = frames

        # 2. Génération des sprites redimensionnés
        self._rescale_assets()

    def _get_cached_zoom_level(self, zoom: float) -> float:
        """Trouve le niveau de zoom caché le plus proche."""
        return min(self.zoom_levels, key=lambda z: abs(z - zoom))

    def _rescale_assets(self):
        """Redimensionne les assets originaux selon self.zoom avec cache."""
        # Snap to nearest cached zoom level for performance
        cached_zoom = self._get_cached_zoom_level(self.zoom)
        
        # Check if we already have this zoom level cached
        if cached_zoom in self._zoom_cache:
            cache = self._zoom_cache[cached_zoom]
            self.grass_sprite = cache.get('grass')
            self.tree_sprite = cache.get('tree')
            self.unit_sprites = cache.get('units', {})
            return
        
        # Create new cache entry
        cache = {}
        
        # Use smoothscale for better quality at low zoom levels
        scale_func = pygame.transform.smoothscale if cached_zoom < 0.6 else pygame.transform.scale
        
        if self.orig_grass:
            w, h = self.orig_grass.get_size()
            cache['grass'] = scale_func(self.orig_grass, (max(1, int(w * cached_zoom)), max(1, int(h * cached_zoom))))
        
        if self.orig_tree:
            w, h = self.orig_tree.get_size()
            cache['tree'] = scale_func(self.orig_tree, (max(1, int(w * cached_zoom)), max(1, int(h * cached_zoom))))

        # Scale unit sprites - utilise la constante globale
        DISPLAY_SCALE = UNIT_DISPLAY_SCALE
        unit_cache = {}
        for u_class, color_dict in self.orig_units.items():
            unit_cache[u_class] = {'blue': {}, 'red': {}}
            for color in ('blue', 'red'):
                for state, frames in (color_dict.get(color) or {}).items():
                    if not frames:
                        unit_cache[u_class][color][state] = None
                        continue
                    scaled_orientations = []
                    target_w = max(1, int(SPRITE_FRAME_WIDTH * DISPLAY_SCALE * cached_zoom))
                    target_h = max(1, int(SPRITE_FRAME_HEIGHT * DISPLAY_SCALE * cached_zoom))
                    for row_frames in frames:
                        scaled_row = []
                        for frame in row_frames:
                            try:
                                scaled = scale_func(frame, (target_w, target_h))
                                scaled_row.append(scaled)
                            except Exception:
                                scaled_row.append(frame)
                        scaled_orientations.append(scaled_row)
                    unit_cache[u_class][color][state] = scaled_orientations
        
        cache['units'] = unit_cache
        
        # Store in cache and apply
        self._zoom_cache[cached_zoom] = cache
        self.grass_sprite = cache.get('grass')
        self.tree_sprite = cache.get('tree')
        self.unit_sprites = unit_cache

    def cart_to_iso(self, x: float, y: float) -> tuple[int, int]:
        """Convertit grille -> isométrique en tenant compte du zoom."""
        iso_x = (x - y) * self.tile_half_w
        iso_y = (x + y) * self.tile_half_h
        final_x = iso_x + self.offset_x - self.scroll_x
        final_y = iso_y + self.offset_y - self.scroll_y
        # Use round() to reduce animation vibration at low zoom
        return round(final_x), round(final_y)

    def iso_to_cart(self, iso_x: float, iso_y: float) -> tuple[float, float]:
        """Convertit écran isométrique -> grille cartésienne (inverse de cart_to_iso)."""
        # Annuler le scroll/offset
        adj_x = iso_x - self.offset_x + self.scroll_x
        adj_y = iso_y - self.offset_y + self.scroll_y
        
        # Système d'équations:
        # iso_x = (x - y) * tile_half_w
        # iso_y = (x + y) * tile_half_h
        # => x - y = adj_x / tile_half_w
        # => x + y = adj_y / tile_half_h
        # => 2x = adj_x/thw + adj_y/thh
        
        term1 = adj_x / self.tile_half_w
        term2 = adj_y / self.tile_half_h
        
        x = (term1 + term2) / 2
        y = (term2 - term1) / 2
        return x, y

    def _clamp_camera(self):
        """Limite la caméra dynamiquement selon la taille de la carte et le zoom."""
        # Calcul de la taille projetée de la carte (demi-diagonales)
        # Largeur totale iso ~ (W + H) * tile_half_w
        # Hauteur totale iso ~ (W + H) * tile_half_h
        
        # On autorise à scroller jusqu'à voir les coins + une marge (écran/2)
        limit_x = (self.map.width + self.map.height) * self.tile_half_w / 2 + self.screen_w * 0.2
        limit_y = (self.map.width + self.map.height) * self.tile_half_h / 2 + self.screen_h * 0.2
        
        self.scroll_x = max(-limit_x, min(self.scroll_x, limit_x))
        self.scroll_y = max(-limit_y, min(self.scroll_y, limit_y))

    def check_events(self) -> str | None:
        """Gère clavier/souris."""
        keys = pygame.key.get_pressed()
        
        # Vitesse de scroll (Maj = rapide, Req 9 PDF)
        current_speed = self.scroll_speed_fast if (keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]) else self.scroll_speed
        
        # Déplacement Caméra (Flèches + ZQSD pour clavier français)
        if keys[pygame.K_LEFT] or keys[pygame.K_q]: self.scroll_x -= current_speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]: self.scroll_x += current_speed
        if keys[pygame.K_UP] or keys[pygame.K_z]: self.scroll_y -= current_speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s] and not (keys[pygame.K_LCTRL] or keys[pygame.K_RCTRL]): 
            self.scroll_y += current_speed
        
        # Limiter la caméra aux bords de la carte
        self._clamp_camera()

        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            
            # --- RESIZE/FULLSCREEN HANDLING ---
            if event.type == pygame.VIDEORESIZE:
                self.screen_w = event.w
                self.screen_h = event.h
                self.screen = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)
                self.update_zoom_metrics()  # Recalculate offsets for new size
            if (event.type == pygame.MOUSEBUTTONDOWN or event.type == pygame.MOUSEMOTION) and \
               pygame.mouse.get_pressed()[0]: # Bouton gauche
                if self.show_minimap:
                    mx, my = pygame.mouse.get_pos()
                    if self._get_minimap_rect().collidepoint(mx, my):
                        gx, gy = self._mm_to_cart(mx, my)
                        self._center_camera_on_grid(gx, gy)
            
            # --- DRAG SOURIS pour scroll (Clic droit + glisser) ---
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:  # Clic droit
                self.is_dragging = True
                self.drag_start_x, self.drag_start_y = event.pos
                self.drag_scroll_start_x = self.scroll_x
                self.drag_scroll_start_y = self.scroll_y
            
            if event.type == pygame.MOUSEBUTTONUP and event.button == 3:
                self.is_dragging = False
            
            if event.type == pygame.MOUSEMOTION and self.is_dragging:
                dx = event.pos[0] - self.drag_start_x
                dy = event.pos[1] - self.drag_start_y
                self.scroll_x = self.drag_scroll_start_x - dx
                self.scroll_y = self.drag_scroll_start_y - dy
                self._clamp_camera()
            
            # --- GESTION DU ZOOM (MOUSEWHEEL) - Centré sur souris ---
            if event.type == pygame.MOUSEWHEEL:
                old_zoom = self.zoom
                # Trouver l'index du niveau de zoom actuel
                current_idx = self.zoom_levels.index(self._get_cached_zoom_level(self.zoom))
                
                if event.y > 0:  # Scroll UP -> Zoom IN (niveau suivant)
                    new_idx = min(current_idx + 1, len(self.zoom_levels) - 1)
                elif event.y < 0:  # Scroll DOWN -> Zoom OUT (niveau précédent)
                    new_idx = max(current_idx - 1, 0)
                else:
                    new_idx = current_idx
                
                new_zoom = self.zoom_levels[new_idx]
                
                if new_zoom != old_zoom:
                    # Position souris à l'écran
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    
                    # Convertir position souris en coordonnées monde (avant zoom)
                    world_x, world_y = self.iso_to_cart(mouse_x, mouse_y)
                    
                    # Appliquer le nouveau zoom
                    self.zoom = new_zoom
                    self.update_zoom_metrics()
                    self._rescale_assets()
                    
                    # Recalculer où cette position monde apparaît maintenant
                    new_screen_x, new_screen_y = self.cart_to_iso(world_x, world_y)
                    
                    # Ajuster le scroll pour garder le point sous la souris
                    self.scroll_x += new_screen_x - mouse_x
                    self.scroll_y += new_screen_y - mouse_y
                    self._clamp_camera()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return "quit"
                if event.key == pygame.K_SPACE: return "toggle_pause"
                if event.key == pygame.K_s: return "step"
                
                # --- Req 12 PDF: F1-F4 Toggle info armée ---
                # F1/F2 capturées par Windows -> alternatives: touches 1 et 2
                if event.key == pygame.K_F1 or event.key == pygame.K_1: 
                    self.show_army_info = not self.show_army_info
                if event.key == pygame.K_F2 or event.key == pygame.K_2: 
                    self.show_hp_bars = not self.show_hp_bars
                if event.key == pygame.K_F3 or event.key == pygame.K_3: 
                    self.show_minimap = not self.show_minimap
                if event.key == pygame.K_F4 or event.key == pygame.K_4: 
                    self.show_unit_details = not self.show_unit_details
                
                # --- Req 11 PDF: M = toggle minimap ---
                if event.key == pygame.K_m:
                    self.show_minimap = not self.show_minimap
                
                # --- Raccourcis F9, F11, F12 (Requis par le PDF) ---
                if event.key == pygame.K_F9: return "switch_view"  # Basculer Terminal/Pygame
                if event.key == pygame.K_F11: return "quick_save"  # Sauvegarde rapide
                if event.key == pygame.K_F12: return "quick_load"  # Chargement rapide

                # --- Req 10 (Variable Speed) ---
                # Claviers portables: PageUp/PageDown (universel) + touches = et 0
                if event.key in (pygame.K_KP_PLUS, pygame.K_PAGEUP, pygame.K_EQUALS): 
                    return "speed_up"
                if event.key in (pygame.K_KP_MINUS, pygame.K_PAGEDOWN, pygame.K_0): 
                    return "speed_down"
        return None

    def draw_map(self):
        """Dessine le sol et les obstacles avec view frustum culling."""
        
        obstacles_dict = {(int(x), int(y)): type_name for type_name, x, y in self.map.obstacles}
        
        # --- VIEW FRUSTUM CULLING ---
        # Calculate visible tile range from screen corners (optimization)
        margin = 100 * self.zoom
        
        # Get cart coords of screen corners
        top_left = self.iso_to_cart(-margin, -margin)
        top_right = self.iso_to_cart(self.screen_w + margin, -margin)
        bottom_left = self.iso_to_cart(-margin, self.screen_h + margin)
        bottom_right = self.iso_to_cart(self.screen_w + margin, self.screen_h + margin)
        
        # Find bounding box in tile coordinates
        all_x = [top_left[0], top_right[0], bottom_left[0], bottom_right[0]]
        all_y = [top_left[1], top_right[1], bottom_left[1], bottom_right[1]]
        
        min_tile_x = max(0, int(min(all_x)) - 1)
        max_tile_x = min(self.map.width, int(max(all_x)) + 2)
        min_tile_y = max(0, int(min(all_y)) - 1)
        max_tile_y = min(self.map.height, int(max(all_y)) + 2)
        
        # Only iterate through visible tiles
        for y in range(min_tile_y, max_tile_y):
            for x in range(min_tile_x, max_tile_x):
                tile = self.map.grid[x][y]
                screen_x, screen_y = self.cart_to_iso(x, y)

                # 1. Calcul de l'élévation (mise à l'échelle)
                height_offset = 0
                if tile.terrain_type != 'water':
                    height_offset = round(tile.elevation * 2 * self.zoom)
                    
                base_y = screen_y - height_offset

                # 2. DESSIN DU FOND (GAZON ou COULEUR)
                if self.grass_sprite and tile.terrain_type != 'water' and tile.elevation == 0:
                    sprite = self.grass_sprite
                    draw_x = screen_x - self.tile_half_w
                    draw_y = screen_y - self.tile_half_h
                    self.screen.blit(sprite, (draw_x, draw_y))
                else:
                    # Fallback : Dessin du losange par couleur d'élévation
                    elev_idx = min(int(tile.elevation / 4), 4)
                    color = TERRAIN_COLORS.get(elev_idx, TERRAIN_COLORS[0])
                    points = [
                        (screen_x, base_y - self.tile_half_h),
                        (screen_x + self.tile_half_w, base_y),
                        (screen_x, base_y + self.tile_half_h),
                        (screen_x - self.tile_half_w, base_y)
                    ]
                    pygame.draw.polygon(self.screen, color, points)

                # 3. Effet 3D (pour les tuiles colorées)
                if height_offset > 0 and tile.terrain_type != 'water':
                     color = TERRAIN_COLORS.get(min(int(tile.elevation / 4), 4), TERRAIN_COLORS[0])
                     darker = (max(0, color[0]-40), max(0, color[1]-40), max(0, color[2]-40))
                     pygame.draw.polygon(self.screen, darker, [
                         (screen_x - self.tile_half_w, base_y),
                         (screen_x, base_y + self.tile_half_h),
                         (screen_x, screen_y + self.tile_half_h),
                         (screen_x - self.tile_half_w, screen_y)
                     ])

                # 4. DESSIN DU SPRITE D'ARBRE
                is_tree = (x, y) in obstacles_dict and obstacles_dict[(x,y)] == "Tree"
                if is_tree and self.tree_sprite:
                    sprite = self.tree_sprite
                    draw_x = screen_x - sprite.get_width() // 2
                    draw_y = screen_y - sprite.get_height()
                    self.screen.blit(sprite, (draw_x, draw_y))


    def draw_units(self, armies: list[Army]):
        """Affiche les unités avec animations extraites des spritesheets.

        Structure attendue : self.unit_sprites[UnitClass][color][state][orientation][frame]
        color = 'blue' pour army_id==0 sinon 'red'.
        """
        visible_units = []
        for army in armies:
            for unit in army.units:
                # Inclure aussi les unités mortes récentes pour afficher l'animation de mort
                visible_units.append((army.army_id, unit))

        # ANTI-VIBRATION: Tri par positions arrondies au dixième + unit_id stable
        # Cela évite les changements d'ordre de dessin quand les positions changent légèrement
        visible_units.sort(key=lambda p: (round(p[1].pos[1], 1), round(p[1].pos[0], 1), p[1].unit_id))
        
        for army_id, unit in visible_units:
            # Ne pas afficher les unités dont l'animation de mort est terminée
            if not unit.is_alive:
                death_elapsed = getattr(unit, 'death_elapsed', 0)
                # Temps max pour l'animation de mort (environ 2 secondes)
                if death_elapsed > 2000:
                    continue  # Skip - l'unité doit disparaître
            
            x, y = unit.pos
            ix, iy = int(x), int(y)
            if not (0 <= ix < self.map.width and 0 <= iy < self.map.height):
                continue

            tile = self.map.grid[ix][iy]
            height_offset = 0
            if tile.terrain_type != 'water':
                height_offset = int(tile.elevation * 2 * self.zoom)

            screen_x, screen_y = self.cart_to_iso(x, y)
            unit_draw_y = screen_y - height_offset

            unit_class = unit.__class__
            sprites_for_unit = self.unit_sprites.get(unit_class)
            color_key = 'blue' if army_id == 0 else 'red'

            # normaliser nom d'état
            state = getattr(unit, 'statut', 'idle')
            if state == 'statique':
                state = 'idle'

            # Si l'animation de mort est marquée comme terminée, ne plus
            # dessiner le sprite (empêche un second cycle visible).
            if getattr(unit, 'death_anim_finished', False):
                continue

            current_frame = None
            if sprites_for_unit:
                frames_for_color = sprites_for_unit.get(color_key, {})
                frames_orient = frames_for_color.get(state)
                if frames_orient:
                    # estimer orientation : privilégier la cible si présente, sinon last_pos
                    # calculer l'angle en espace écran (isométrique) pour correspondre à l'orientation des sprites
                    tgt = None
                    target_id = getattr(unit, 'target_id', None)
                    if target_id is not None:
                        for a in armies:
                            for uu in a.units:
                                if uu.unit_id == target_id:
                                    tgt = uu
                                    break
                            if tgt: break

                    if tgt is not None:
                        u_xs, u_ys = self.cart_to_iso(unit.pos[0], unit.pos[1])
                        t_xs, t_ys = self.cart_to_iso(tgt.pos[0], tgt.pos[1])
                        vx = t_xs - u_xs
                        vy = t_ys - u_ys
                    else:
                        lx, ly = getattr(unit, 'last_pos', unit.pos)
                        u_xs, u_ys = self.cart_to_iso(unit.pos[0], unit.pos[1])
                        l_xs, l_ys = self.cart_to_iso(lx, ly)
                        vx = u_xs - l_xs
                        vy = u_ys - l_ys

                    # Calculer l'orientation basée sur la direction
                    rows = len(frames_orient)
                    
                    # Calculer l'orientation basée sur vx/vy
                    if abs(vx) > 2.0 or abs(vy) > 2.0:
                        angle = math.degrees(math.atan2(vy, vx)) % 360
                        sector = 360.0 / max(1, rows)
                        orient_idx = int((angle + sector/2) // sector) % rows
                        unit._last_orient = orient_idx
                    else:
                        # Pas de mouvement significatif
                        if hasattr(unit, '_last_orient'):
                            orient_idx = unit._last_orient % rows
                        else:
                            # Orientation initiale vers le centre de la carte
                            map_center_x, map_center_y = self.map.width / 2, self.map.height / 2
                            cx, cy = self.cart_to_iso(map_center_x, map_center_y)
                            ux, uy = self.cart_to_iso(unit.pos[0], unit.pos[1])
                            dx, dy = cx - ux, cy - uy
                            if abs(dx) > 0.1 or abs(dy) > 0.1:
                                angle = math.degrees(math.atan2(dy, dx)) % 360
                                sector = 360.0 / max(1, rows)
                                orient_idx = int((angle + sector/2) // sector) % rows
                            else:
                                # Keep last orientation if not moving
                                orient_idx = getattr(unit, '_last_orient', 0)
                            unit._last_orient = orient_idx

                    nframes = len(frames_orient[orient_idx]) if frames_orient[orient_idx] else 0
                    if nframes > 0:
                        if state == 'death' and not unit.is_alive:
                            idx = min(getattr(unit, 'anim_index', 0), nframes - 1)
                        else:
                            idx = getattr(unit, 'anim_index', 0) % nframes
                        try:
                            current_frame = frames_orient[orient_idx][idx]
                        except Exception:
                            current_frame = None

            if current_frame:
                surf = current_frame
                draw_x = screen_x - surf.get_width() // 2
                draw_y = unit_draw_y - surf.get_height()
                self.screen.blit(surf, (draw_x, draw_y))
                hp_y = unit_draw_y - surf.get_height() + 5
            else:
                # fallback minimal: dessiner seulement si l'unité est vivante
                if unit.is_alive:
                    color = BLUE if army_id == 0 else RED
                    
                    # Logique de dessin de fallback (Cercle vs Carré)
                    # Si c'est un Bâtiment (hitbox > 0.6 ou type spécifique), on fait un carré/rectangle
                    is_building = unit.hitbox_radius > 0.6
                    
                    if is_building:
                        # Dessin de bâtiment (Carré 3D simple)
                        size = int(unit.hitbox_radius * 2 * 32 * self.zoom) # 32 pixels/mètre approx
                        rect_x = screen_x - size // 2
                        rect_y = unit_draw_y - size
                        
                        # Couleur spécifique pour Castle/Wonder
                        if isinstance(unit, Castle):
                            draw_color = (150, 150, 150) # Gris Pierre
                        elif isinstance(unit, Wonder):
                            draw_color = (255, 215, 0) # Or
                        else:
                            draw_color = color
                            
                        # Face avant
                        pygame.draw.rect(self.screen, draw_color, (rect_x, rect_y, size, size))
                        # Bordure
                        pygame.draw.rect(self.screen, BLACK, (rect_x, rect_y, size, size), 2)
                        
                        # Marque de couleur du joueur
                        mark_size = size // 4
                        pygame.draw.rect(self.screen, color, (rect_x + size//2 - mark_size//2, rect_y + size//4, mark_size, mark_size))

                        hp_y = rect_y - 10
                    else:
                        # Dessin d'unité standard (Cercle)
                        radius = max(8, int(10 * self.zoom))
                        circle_y = unit_draw_y - int(15 * self.zoom)
                        pygame.draw.circle(self.screen, color, (screen_x, circle_y), radius)
                        pygame.draw.circle(self.screen, WHITE, (screen_x, circle_y), radius, 2)
                        hp_y = circle_y - radius - 5
                else:
                    # unité morte et pas de sprite: ne rien dessiner
                    hp_y = unit_draw_y

            if self.show_hp_bars and unit.is_alive:
                hp_ratio = unit.current_hp / unit.max_hp if unit.max_hp > 0 else 0
                bar_width = int(16 * self.zoom)  # Plus petit
                bar_height = max(2, int(2 * self.zoom))  # Plus fin
                bar_y = hp_y + 8  # Plus proche du perso
                pygame.draw.rect(self.screen, RED, (screen_x - bar_width//2, bar_y, bar_width, bar_height))
                pygame.draw.rect(self.screen, GREEN, (screen_x - bar_width//2, bar_y, int(bar_width * hp_ratio), bar_height))


    def draw_ui(self, time_elapsed, paused, armies):
        """Interface Utilisateur avec toggles F1-F4 - Design moderne."""
        
        # --- TIME DISPLAY (Top center - always visible) ---
        minutes = int(time_elapsed) // 60
        seconds = int(time_elapsed) % 60
        time_str = f"Duree: {minutes:02d}:{seconds:02d}"
        
        time_surface = pygame.Surface((160, 35), pygame.SRCALPHA)
        time_surface.fill((15, 15, 20, 200))
        pygame.draw.rect(time_surface, (80, 80, 90), (0, 0, 160, 35), 1)
        time_txt = self.ui_font.render(time_str, True, (220, 220, 220))
        time_surface.blit(time_txt, (time_surface.get_width()//2 - time_txt.get_width()//2, 5))
        self.screen.blit(time_surface, (self.screen_w//2 - 80, 10))
        
        # --- PAUSE OVERLAY WITH FULL CONTROLS ---
        if paused:
            # Semi-transparent overlay
            overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            self.screen.blit(overlay, (0, 0))
            
            # Main pause panel
            panel_w, panel_h = 600, 550
            panel_x = self.screen_w // 2 - panel_w // 2
            panel_y = self.screen_h // 2 - panel_h // 2
            
            pause_panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            pause_panel.fill((20, 22, 28, 240))
            pygame.draw.rect(pause_panel, (70, 130, 220), (0, 0, panel_w, panel_h), 3)
            
            # Title
            title_font = pygame.font.SysFont('Arial', 36, bold=True)
            title = title_font.render("PAUSE", True, (255, 200, 100))
            pause_panel.blit(title, (panel_w//2 - title.get_width()//2, 15))
            
            # Separator
            pygame.draw.line(pause_panel, (60, 60, 70), (25, 60), (panel_w - 25, 60), 2)
            
            # Controls sections (SIMPLIFIED)
            controls = [
                ("NAVIGATION", [
                    ("Fleches / ZQSD / Clic droit", "Deplacer camera"),
                    ("Molette souris", "Zoom +/-"),
                    ("Maj + fleches", "Deplacement rapide"),
                ]),
                ("JEU", [
                    ("Espace", "Pause / Reprendre"),
                    ("= / à", "Accelerer / Ralentir"),
                    ("+ / -", "Accelerer / Ralentir (sur pavé numérique)"),
                    ("Echap", "Quitter"),
                ]),
                ("AFFICHAGE", [
                    ("1 / F1", f"Infos armee {'[ON]' if self.show_army_info else '[OFF]'}"),
                    ("2 / F2", f"Barres HP {'[ON]' if self.show_hp_bars else '[OFF]'}"),
                    ("3 / F3 / M", f"Minimap {'[ON]' if self.show_minimap else '[OFF]'}"),
                    ("4 / F4", f"Details unites {'[ON]' if self.show_unit_details else '[OFF]'}"),
                ]),
                ("SAUVEGARDE", [
                    ("F11", "Sauvegarde rapide"),
                    ("F12", "Charger sauvegarde"),
                ]),
            ]
            
            y_offset = 75
            section_font = pygame.font.SysFont('Arial', 20, bold=True)
            control_font = pygame.font.SysFont('Arial', 17)
            
            for section_name, section_controls in controls:
                # Section header (BIGGER)
                section_txt = section_font.render(section_name, True, (150, 180, 220))
                pause_panel.blit(section_txt, (25, y_offset))
                y_offset += 28
                
                for key, desc in section_controls:
                    # Key (BIGGER)
                    key_txt = control_font.render(key, True, (200, 200, 200))
                    pause_panel.blit(key_txt, (40, y_offset))
                    # Description
                    desc_txt = control_font.render(desc, True, (140, 140, 150))
                    pause_panel.blit(desc_txt, (260, y_offset))
                    y_offset += 22
                
                y_offset += 10  # Space between sections
            
            # Resume hint at bottom
            resume_txt = self.font.render("Appuyez sur ESPACE pour reprendre", True, (100, 200, 100))
            pause_panel.blit(resume_txt, (panel_w//2 - resume_txt.get_width()//2, panel_h - 30))
            
            self.screen.blit(pause_panel, (panel_x, panel_y))
        
        # --- F1: ARMY INFO PANELS (Beautiful redesign) ---
        if self.show_army_info:
            # Position armies at top corners for VS-style display
            for i, army in enumerate(armies):
                alive = sum(1 for u in army.units if u.is_alive)
                total = getattr(army, 'initial_count', len(army.units))
                percent = (alive / total * 100) if total > 0 else 0
                total_hp = sum(u.current_hp for u in army.units if u.is_alive)
                max_total_hp = getattr(army, 'initial_total_hp', sum(u.max_hp for u in army.units))
                hp_percent = (total_hp / max_total_hp) if max_total_hp > 0 else 0
                general_name = army.general.__class__.__name__
                
                # Panel dimensions
                panel_w, panel_h = 260, 85
                
                # Utiliser army.army_id pour correctement associer position et couleur
                team = army.army_id  # 0 = bleu (gauche), 1 = rouge (droite)
                
                # Position: right for army 0 (blue), left for army 1 (red) - correspond à la carte
                if team == 0:
                    panel_x = self.screen_w - panel_w - 15  # Bleu à droite
                else:
                    panel_x = 15  # Rouge à gauche
                panel_y = 10
                
                # Team colors with glow effect (basé sur army_id, pas index)
                if team == 0:
                    main_color = (50, 120, 200)
                    glow_color = (80, 150, 255)
                    bar_color = (100, 180, 255)
                else:
                    main_color = (200, 60, 60)
                    glow_color = (255, 100, 100)
                    bar_color = (255, 120, 120)
                
                # Panel with gradient-like effect
                panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
                
                # Background gradient simulation (darker at top, lighter at bottom)
                for y in range(panel_h):
                    alpha = 200 + int(40 * (y / panel_h))
                    shade = 15 + int(10 * (y / panel_h))
                    pygame.draw.line(panel, (shade, shade, shade + 5, min(255, alpha)), (0, y), (panel_w, y))
                
                # Glowing border on team side
                border_side = panel_w - 5 if team == 0 else 0
                for glow in range(5):
                    alpha = 150 - glow * 30
                    pygame.draw.rect(panel, (*glow_color, max(0, alpha)), (border_side + (glow if team == 0 else -glow), 0, 5 - glow, panel_h))
                
                # Main team color bar
                pygame.draw.rect(panel, main_color, (panel_w - 5 if team == 0 else 0, 0, 5, panel_h))
                
                # Outer border
                pygame.draw.rect(panel, (60, 60, 70), (0, 0, panel_w, panel_h), 1)
                
                # Army name and general
                name_x = 15 if team == 0 else 10
                title = self.ui_font.render(f"ARMEE {team+1}", True, glow_color)
                panel.blit(title, (name_x, 8))
                
                general_txt = self.font.render(general_name, True, (180, 180, 180))
                panel.blit(general_txt, (name_x, 32))
                
                # Formation status (Req)
                formation_mode = getattr(army.general, 'formation_mode', '')
                if formation_mode:
                    fmt_txt = self.font.render(f"     Strat: {formation_mode}", True, (150, 200, 150))
                    panel.blit(fmt_txt, (name_x + 90, 32))
                
                # Stats row
                stats_y = 52
                
                # Units alive
                units_str = f"{alive}/{total} unites"
                units_txt = self.font.render(units_str, True, WHITE)
                panel.blit(units_txt, (name_x, stats_y))
                
                # HP with colored bar (Vie globale)
                vie_label = self.font.render("Vie:", True, (150, 150, 150))
                panel.blit(vie_label, (name_x + 95, stats_y))
                
                hp_bar_x = name_x + 125
                hp_bar_w = 85
                hp_bar_h = 14
                
                # HP bar background
                pygame.draw.rect(panel, (30, 30, 35), (hp_bar_x, stats_y, hp_bar_w, hp_bar_h))
                # HP bar fill
                pygame.draw.rect(panel, bar_color, (hp_bar_x, stats_y, int(hp_bar_w * hp_percent), hp_bar_h))
                # HP bar border
                pygame.draw.rect(panel, (80, 80, 90), (hp_bar_x, stats_y, hp_bar_w, hp_bar_h), 1)
                
                # HP percentage text on bar
                hp_pct_txt = self.font.render(f"{int(hp_percent * 100)}%", True, WHITE)
                panel.blit(hp_pct_txt, (hp_bar_x + hp_bar_w//2 - hp_pct_txt.get_width()//2, stats_y - 1))
                
                self.screen.blit(panel, (panel_x, panel_y))
        
        # --- F4: UNIT DETAILS PANELS (positioned under respective army) ---
        if self.show_unit_details:
            detail_panel_width = 180
            
            for i, army in enumerate(armies):
                team = army.army_id  # Utiliser army_id pas l'index
                team_color = (70, 130, 220) if team == 0 else (220, 70, 70)
                accent_color = (100, 160, 255) if team == 0 else (255, 100, 100)
                
                # Count units by type
                unit_counts = {}
                initial_breakdown = getattr(army, 'initial_units_breakdown', {})
                
                # Compter les vivants actuels
                current_alive = {}
                for u in army.units:
                    if u.is_alive:
                         name = u.__class__.__name__
                         current_alive[name] = current_alive.get(name, 0) + 1
                
                # Fusionner avec les totaux initiaux
                all_types = set(initial_breakdown.keys()) | set(current_alive.keys())
                
                for name in all_types:
                    total = initial_breakdown.get(name, 0)
                    # Fallback si pas de stats initiales (ex: unités spawnées ou vieux save)
                    if total == 0:
                         total = sum(1 for u in army.units if u.__class__.__name__ == name)
                    
                    alive = current_alive.get(name, 0)
                    unit_counts[name] = {'alive': alive, 'total': total}
                
                # Calculate panel height
                num_types = len(unit_counts)
                detail_panel_height = 28 + num_types * 18
                
                # Position: right for army 0 (blue), left for army 1 (red)
                if team == 0:
                    detail_x = self.screen_w - detail_panel_width - 15
                else:
                    detail_x = 15
                detail_y = 100  # Under army panels
                
                # Panel background
                panel = pygame.Surface((detail_panel_width, detail_panel_height), pygame.SRCALPHA)
                panel.fill((20, 20, 25, 200))
                pygame.draw.rect(panel, (40, 40, 50), (0, 0, detail_panel_width, detail_panel_height), 1)
                border_x = detail_panel_width - 3 if team == 0 else 0
                pygame.draw.rect(panel, team_color, (border_x, 0, 3, detail_panel_height))
                
                # Header
                header = self.font.render(f"Details", True, accent_color)
                panel.blit(header, (10 if i == 0 else 8, 5))
                
                # Unit type rows
                row_y = 24
                
                for unit_type, counts in unit_counts.items():
                    alive, total = counts['alive'], counts['total']
                    ratio = alive / total if total > 0 else 0
                    
                    # Status color
                    if ratio == 1:
                        status_color = (100, 200, 100)
                    elif ratio > 0.5:
                        status_color = (200, 200, 100)
                    elif ratio > 0:
                        status_color = (200, 100, 100)
                    else:
                        status_color = (100, 100, 100)
                    
                    short_name = unit_type[:10] if len(unit_type) > 10 else unit_type
                    line = self.font.render(f"{short_name}: {alive}/{total}", True, status_color)
                    panel.blit(line, (10 if i == 0 else 8, row_y))
                    row_y += 16
                
                self.screen.blit(panel, (detail_x, detail_y))
        
        # --- MINI CONTROLS HINT (Bottom right - only when NOT paused) ---
        if not paused:
            hint_surface = pygame.Surface((180, 25), pygame.SRCALPHA)
            hint_surface.fill((0, 0, 0, 120))
            hint_txt = self.font.render("Espace = Pause / Controles", True, (140, 140, 140))
            hint_surface.blit(hint_txt, (8, 4))
            self.screen.blit(hint_surface, (self.screen_w - 195, self.screen_h - 35))

        # Minimap (F3/M toggle)
        if self.show_minimap:
            p1 = self._cart_to_mm(0, 0)
            p2 = self._cart_to_mm(self.map.width, 0)
            p3 = self._cart_to_mm(self.map.width, self.map.height)
            p4 = self._cart_to_mm(0, self.map.height)
            
            pygame.draw.polygon(self.screen, (20, 20, 20), [p1, p2, p3, p4])
            pygame.draw.polygon(self.screen, (100, 100, 100), [p1, p2, p3, p4], 2)

            for army in armies:
                c = BLUE if army.army_id == 0 else RED
                for u in army.units:
                    if u.is_alive:
                        mx, my = self._cart_to_mm(u.pos[0], u.pos[1])
                        pygame.draw.circle(self.screen, c, (mx, my), 2)

            corners_screen = [(0, 0), (self.screen_w, 0), (self.screen_w, self.screen_h), (0, self.screen_h)]
            poly_points = []
            for sx, sy in corners_screen:
                gx, gy = self.iso_to_cart(sx, sy)
                gx, gy = max(0, min(self.map.width, gx)), max(0, min(self.map.height, gy))
                poly_points.append(self._cart_to_mm(gx, gy))
            
            if len(poly_points) == 4:
                pygame.draw.polygon(self.screen, WHITE, poly_points, 1)

    def display(self, armies: list[Army], time_elapsed: float, paused: bool = False, speed_multiplier: float = 1.0) -> str | None:
        cmd = self.check_events()

        # Avancer les animations par frame (delta constant pour fluidité)
        # Utiliser un delta constant de 16ms (environ 60 FPS) pour des animations stables
        anim_delta = 16  # ms constant pour des animations fluides
        
        # Si le jeu est en pause, ne pas avancer les animations
        if not paused:
            try:
                for army in armies:
                    for unit in army.units:
                        try:
                            unit.tick_animation(anim_delta)
                        except Exception:
                            pass
            except Exception:
                pass

        self.screen.fill(BG_COLOR)
        self.draw_map()
        self.draw_units(armies)
        self.draw_ui(time_elapsed, paused, armies)
        pygame.display.flip()
        self.clock.tick(60)
        return cmd

    def display_game_over(self, armies: list[Army], winner_id: int | None, time_elapsed: float):
        """Affiche l'écran de fin de partie avec animations et stats."""
        
        # Calcul des stats détaillées
        stats = []
        for army in armies:
            alive_units = [u for u in army.units if u.is_alive]
            dead_units = [u for u in army.units if not u.is_alive]
            alive = len(alive_units)
            total = getattr(army, 'initial_count', len(army.units))
            hp_remaining = sum(u.current_hp for u in alive_units)
            max_hp = getattr(army, 'initial_total_hp', sum(u.max_hp for u in army.units))
            hp_lost = max_hp - hp_remaining
            
            # Stats par type d'unité
            unit_types = {}
            initial_bd = getattr(army, 'initial_units_breakdown', {})
            
            # Initialiser avec les totaux
            for name, count in initial_bd.items():
                unit_types[name] = {'alive': 0, 'dead': count, 'hp': 0, 'max_hp': 0}

            for u in army.units:
                name = u.__class__.__name__
                if name not in unit_types:
                    unit_types[name] = {'alive': 0, 'dead': 0, 'hp': 0, 'max_hp': 0}
                
                unit_types[name]['max_hp'] += u.max_hp
                if u.is_alive:
                    unit_types[name]['alive'] += 1
                    unit_types[name]['hp'] += u.current_hp
                    if name in initial_bd:
                        unit_types[name]['dead'] = initial_bd[name] - unit_types[name]['alive'] 
                else: 
                     # Si l'unité morte est encore dans la liste (ex: animation mort non finie)
                     # Le dead count est déjà géré par l'initialisation sauf si hors initial_bd
                     if name not in initial_bd:
                        unit_types[name]['dead'] += 1
            
            stats.append({
                'army_id': army.army_id,  # Pour la couleur correcte
                'name': f"Armee {army.army_id + 1}",
                'general': army.general.__class__.__name__,
                'alive': alive,
                'total': total,
                'hp': hp_remaining,
                'max_hp': max_hp,
                'hp_lost': hp_lost,
                'survival_rate': (alive / total * 100) if total > 0 else 0,
                'hp_rate': (hp_remaining / max_hp * 100) if max_hp > 0 else 0,
                'unit_types': unit_types,
            })
        
        start_time = pygame.time.get_ticks()
        
        while True:
            elapsed = pygame.time.get_ticks() - start_time
            progress = min(1.0, elapsed / 600)  # Animation fade-in en 0.6s
            
            # Gestion des événements
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    return
                if event.type == pygame.MOUSEBUTTONDOWN:
                    return
            
            # Dessiner le fond
            self.screen.fill(BG_COLOR)
            self.draw_map()
            self.draw_units(armies)
            
            # Overlay sombre avec fade-in
            overlay_alpha = min(200, int(220 * progress))
            overlay = pygame.Surface((self.screen_w, self.screen_h), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, overlay_alpha))
            self.screen.blit(overlay, (0, 0))
            
            # Panel central plus grand
            panel_w, panel_h = 650, 480
            panel_x = self.screen_w // 2 - panel_w // 2
            panel_y = self.screen_h // 2 - panel_h // 2
            
            # Animation de slide
            slide_offset = int((1 - progress) * 80)
            panel_y += slide_offset
            
            panel = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
            
            # Fond avec dégradé simulé
            for y in range(panel_h):
                shade = 12 + int(8 * (y / panel_h))
                pygame.draw.line(panel, (shade, shade + 2, shade + 5, 245), (0, y), (panel_w, y))
            
            # Bordure extérieure
            pygame.draw.rect(panel, (60, 65, 80), (0, 0, panel_w, panel_h), 2)
            
            # Titre avec couleur de l'équipe gagnante
            if winner_id is not None:
                winner_team_color = (100, 170, 255) if winner_id == 0 else (255, 110, 110)
                title_text = f"VICTOIRE DE L'ARMEE {winner_id + 1}"
                title_color = winner_team_color
            else:
                title_text = "EGALITE"
                title_color = (180, 180, 180)
            
            # Titre animé - plus grand
            title_scale = 0.6 + 0.4 * min(1.0, progress * 2)
            title_font = pygame.font.SysFont('Arial', int(32 * title_scale), bold=True)
            title = title_font.render(title_text, True, title_color)
            panel.blit(title, (panel_w // 2 - title.get_width() // 2, 15))
            
            # Durée de la bataille - plus visible
            minutes = int(time_elapsed) // 60
            seconds = int(time_elapsed) % 60
            time_str = f"Duree: {minutes}m {seconds}s"
            time_font = pygame.font.SysFont('Arial', 20, bold=True)
            time_txt = time_font.render(time_str, True, (255, 215, 100))
            panel.blit(time_txt, (panel_w // 2 - time_txt.get_width() // 2, 50))
            
            # Ligne de séparation
            pygame.draw.line(panel, (50, 55, 70), (25, 80), (panel_w - 25, 80), 1)
            
            # === SECTION STATS - 2 colonnes avec plus d'espace ===
            gap = 60  # Espace entre les colonnes pour le VS
            col_w = (panel_w - 50 - gap) // 2
            
            for i, s in enumerate(stats):
                team = s['army_id']  # Utiliser army_id pas l'index
                # Position de la colonne - plus d'espace au milieu
                col_x = panel_w - col_w - 25 if team == 0 else 25
                base_y = 90
                
                # Couleurs équipe (basées sur army_id)
                if team == 0:
                    team_color = (80, 140, 220)
                    accent = (100, 170, 255)
                    bar_color = (70, 140, 220)
                else:
                    team_color = (220, 80, 80)
                    accent = (255, 110, 110)
                    bar_color = (220, 70, 70)
                
                is_winner = (winner_id == team)
                
                # Fond de colonne légèrement coloré
                col_bg = pygame.Surface((col_w, 370), pygame.SRCALPHA)
                col_bg.fill((*team_color, 15))
                panel.blit(col_bg, (col_x, base_y))
                
                # Bordure latérale colorée
                pygame.draw.rect(panel, team_color, (col_x, base_y, 4, 370))
                
                # === EN-TETE ===
                header_y = base_y + 8
                
                winner_icon = "" if is_winner else ""
                name_txt = self.ui_font.render(f"{winner_icon}{s['name']}", True, accent)
                panel.blit(name_txt, (col_x + 12, header_y))

                
                # Général
                gen_txt = self.font.render(f"General: {s['general']}", True, (150, 150, 160))
                panel.blit(gen_txt, (col_x + 12, header_y + 26))
                
                # === STATS GLOBALES ===
                stat_y = header_y + 55
                
                # Taux de survie
                survival_label = self.font.render("Taux de survie:", True, (130, 130, 140))
                panel.blit(survival_label, (col_x + 12, stat_y))
                survival_val = self.font.render(f"{s['survival_rate']:.0f}%", True, (200, 200, 200))
                panel.blit(survival_val, (col_x + col_w - 50, stat_y))
                
                # Barre de survie
                bar_y = stat_y + 18
                bar_w = col_w - 24
                pygame.draw.rect(panel, (30, 30, 35), (col_x + 12, bar_y, bar_w, 8))
                pygame.draw.rect(panel, bar_color, (col_x + 12, bar_y, int(bar_w * s['survival_rate'] / 100), 8))
                
                # Unités survivantes (simplifié)
                unit_y = bar_y + 16
                units_txt = self.font.render(f"{s['alive']}/{s['total']} unites", True, (170, 170, 180))
                panel.blit(units_txt, (col_x + 12, unit_y))
                
                # HP Global
                hp_y = unit_y + 22
                hp_label = self.font.render("Vie globale:", True, (130, 130, 140))
                panel.blit(hp_label, (col_x + 12, hp_y))
                hp_val = self.font.render(f"{s['hp_rate']:.0f}%", True, (200, 200, 200))
                panel.blit(hp_val, (col_x + col_w - 50, hp_y))
                
                # Barre HP
                hp_bar_y = hp_y + 18
                pygame.draw.rect(panel, (30, 30, 35), (col_x + 12, hp_bar_y, bar_w, 8))
                hp_fill = (80, 180, 80) if s['hp_rate'] > 50 else ((200, 180, 60) if s['hp_rate'] > 25 else (200, 80, 80))
                pygame.draw.rect(panel, hp_fill, (col_x + 12, hp_bar_y, int(bar_w * s['hp_rate'] / 100), 8))
                
                # HP chiffré
                hp_num_y = hp_bar_y + 14
                hp_num = self.font.render(f"{s['hp']}/{s['max_hp']} HP", True, (140, 140, 150))
                panel.blit(hp_num, (col_x + 12, hp_num_y))
                
                
                # === DETAILS PAR TYPE ===
                type_y = hp_num_y + 28
                type_header = self.font.render("Details des survivants par unites:", True, (120, 120, 130))
                panel.blit(type_header, (col_x + 12, type_y))
                
                type_y += 20
                for unit_name, ut in s['unit_types'].items():
                    # Nom + stats
                    status_color = (100, 180, 100) if ut['dead'] == 0 else ((180, 180, 80) if ut['alive'] > 0 else (150, 80, 80))
                    ut_txt = self.font.render(f"{unit_name}: {ut['alive']}/{ut['alive']+ut['dead']}", True, status_color)
                    panel.blit(ut_txt, (col_x + 16, type_y))
                    

                    
                    type_y += 18
            
            # VS au centre - bien centré entre les colonnes
            vs_y = 200
            vs_x = panel_w // 2
            vs_bg = pygame.Surface((50, 50), pygame.SRCALPHA)
            pygame.draw.circle(vs_bg, (40, 40, 50, 200), (25, 25), 25)
            panel.blit(vs_bg, (vs_x - 25, vs_y))
            vs_txt = self.ui_font.render("VS", True, (100, 100, 120))
            panel.blit(vs_txt, (vs_x - vs_txt.get_width() // 2, vs_y + 12))
            
            # === BAS DU PANEL ===
            # Hint
            hint = self.font.render("Appuyez sur une touche pour quitter...", True, (90, 90, 100))
            panel.blit(hint, (panel_w // 2 - hint.get_width() // 2, panel_h - 25))
            
            self.screen.blit(panel, (panel_x, panel_y))
            
            pygame.display.flip()
            self.clock.tick(60)

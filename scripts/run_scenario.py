# scripts/run_scenario.py
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
from core.army import Army
from core.unit import Unit
from ai.generals import MajorDAFT
from core.definitions import GENERAL_CLASS_MAP, UNIT_CLASS_MAP

def lanchester_scenario(unit_class: type[Unit], n: int, general_class=MajorDAFT):
    """
    Creates a scenario for testing Lanchester's Laws.
    """
    comp1 = {unit_class.__name__: n}
    comp2 = {unit_class.__name__: 2*n}
    
    return custom_battle_scenario(
        comp1, comp2, 
        general_class, general_class, 
        (40, 40)
    )

def custom_battle_scenario(comp1: dict[str, int], 
                           comp2: dict[str, int],
                           general_class1: type, 
                           general_class2: type,
                           map_size: tuple[int, int] = (120, 120)):
    
    width, height = map_size
    # FIX 1: Increased spacing to 2.5 to accommodate Elephants and prevent "stuck" collisions
    spacing = 2.5 
    
    def get_unit_range(name: str) -> float:
        cls = UNIT_CLASS_MAP.get(name)
        if not cls: return 0.0
        dummy = cls(unit_id=0, army_id=0, pos=(0, 0))
        return dummy.attack_range

    # Helper to calculate block dimensions
    def get_block_dims(total_n):
        if total_n <= 0: return 1, 0, 0
        cols = int(math.ceil(math.sqrt(total_n)))
        rows = math.ceil(total_n / cols)
        return cols, (cols - 1) * spacing, (rows - 1) * spacing

    # FIX 2: Dynamic half_gap that respects map boundaries
    cols1, block_w1, block_h1 = get_block_dims(sum(comp1.values()))
    cols2, block_w2, block_h2 = get_block_dims(sum(comp2.values()))
    
    # Ensure a 5-unit margin from the top and bottom edges
    max_block_h = max(block_h1, block_h2)
    safe_half_gap = min(height / 2 - 5, height / 2 - max_block_h - 5)
    half_gap = max(5, safe_half_gap) if height < 90 else 35

    center_x, center_y = width / 2, height / 2
    
    # --- ARMEE 1 (TOP/BLUE) ---
    army1_units = []
    start_x1 = center_x - block_w1 / 2
    start_y1 = max(2, center_y - half_gap - block_h1) # Avoid Y=0 clamp
    
    sorted_types1 = sorted(comp1.keys(), key=get_unit_range, reverse=True)
    current_idx = 0
    for unit_name in sorted_types1:
        u_class = UNIT_CLASS_MAP[unit_name]
        for _ in range(comp1[unit_name]):
            row, col = divmod(current_idx, cols1)
            # Front line is the highest row index
            pos_x = start_x1 + col * spacing
            pos_y = start_y1 + row * spacing
            army1_units.append(u_class(unit_id=current_idx, army_id=0, pos=(pos_x, pos_y)))
            current_idx += 1

    # --- ARMEE 2 (BOTTOM/RED) ---
    army2_units = []
    start_x2 = center_x + block_w2 / 2
    start_y2 = min(height - 2, center_y + half_gap + block_h2) # Avoid Y=height clamp

    sorted_types2 = sorted(comp2.keys(), key=get_unit_range, reverse=True)
    current_idx_2 = 0
    for unit_name in sorted_types2:
        u_class = UNIT_CLASS_MAP[unit_name]
        for _ in range(comp2[unit_name]):
            row, col = divmod(current_idx_2, cols2)
            pos_x = start_x2 - col * spacing
            pos_y = start_y2 - row * spacing
            army2_units.append(u_class(unit_id=10000 + current_idx_2, army_id=1, pos=(pos_x, pos_y)))
            current_idx_2 += 1

    return Army(0, army1_units, general_class1(0)), Army(1, army2_units, general_class2(1))

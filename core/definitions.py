# core/definitions.py
from ai.generals import CaptainBRAINDEAD, MajorDAFT, ColonelKAISER
from core.unit import (
    Knight, Pikeman, Crossbowman, LongSwordsman,
    EliteSkirmisher, CavalryArcher, Onager, Castle, Wonder,
    LightCavalry, Scorpion, CappedRam, Trebuchet, EliteWarElephant, Monk
)

GENERAL_CLASS_MAP = {
    "CaptainBRAINDEAD": CaptainBRAINDEAD,
    "MajorDAFT": MajorDAFT,
    "ColonelKAISER": ColonelKAISER,
}

UNIT_CLASS_MAP = {
    # Unités de base
    "Knight": Knight,
    "Pikeman": Pikeman,
    "Crossbowman": Crossbowman,
    "LongSwordsman": LongSwordsman,
    "EliteSkirmisher": EliteSkirmisher,
    "CavalryArcher": CavalryArcher,
    "Onager": Onager,
    # Nouvelles unités (Req PDF)
    "LightCavalry": LightCavalry,
    "Scorpion": Scorpion,
    "CappedRam": CappedRam,
    "Trebuchet": Trebuchet,
    "EliteWarElephant": EliteWarElephant,
    "Monk": Monk,
    # Bâtiments
    "Castle": Castle,
    "Wonder": Wonder,
}

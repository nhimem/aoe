# core/army.py
from core.unit import Unit, UC_BUILDING
from ai.general import General

class Army:
    """
    Représente une armée (req 4).
    Contient une liste d'unités et le Général (IA) qui les commande.
    """
    def __init__(self, army_id: int, units: list[Unit], general: General):
        self.army_id: int = army_id
        self.units: list[Unit] = units
        self.general: General = general

        # Initial stats storage for accurate UI percentages
        self.initial_count = len(units)
        self.initial_total_hp = sum(u.max_hp for u in units)
        self.initial_units_breakdown = {}
        for u in units:
            name = u.__class__.__name__
            self.initial_units_breakdown[name] = self.initial_units_breakdown.get(name, 0) + 1

        # S'assure que le général est bien lié à cette armée
        self.general.army_id = army_id

    def __repr__(self) -> str:
        return f"Army({self.army_id}, Général: {self.general.__class__.__name__}, Unités: {len(self.units)})"

    def is_defeated(self) -> bool:
        """
        Vérifie si l'armée est vaincue.
        Une armée est vaincue si elle n'a plus d'unités mobiles (non-bâtiments) en vie.
        (Exclut les bâtiments pour éviter les parties infinies contre des murs/châteaux)
        """
        if not self.units:
            return True # Une armée sans unité est vaincue

        # On cherche s'il reste au moins une unité vivante qui n'est PAS un bâtiment
        has_living_mobile_unit = any(
            unit.is_alive and UC_BUILDING not in unit.armor_classes
            for unit in self.units
        )

        return not has_living_mobile_unit

    def to_dict(self) -> dict:
        """Sérialise l'armée en dictionnaire."""
        return {
            'army_id': self.army_id,
            'general_type': self.general.__class__.__name__,
            'units': [u.to_dict() for u in self.units],
            'initial_stats': {
                'count': self.initial_count,
                'hp': self.initial_total_hp,
                'breakdown': self.initial_units_breakdown
            }
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Army':
        """Reconstruit une armée depuis un dictionnaire."""
        from core.definitions import GENERAL_CLASS_MAP
        
        army_id = data['army_id']
        general_type = data['general_type']
        
        # Reconstruire le Général
        gen_class = GENERAL_CLASS_MAP.get(general_type)
        if not gen_class:
            # Fallback si l'IA n'existe plus ou mismatch
            print(f"Attention: Général '{general_type}' inconnu, fallback sur MajorDAFT")
            from ai.generals import MajorDAFT
            gen_class = MajorDAFT
            
        general = gen_class(army_id)
        
        # Reconstruire les unités
        units = [Unit.from_dict(u_data) for u_data in data['units']]
        
        # Créer l'armée (cela recalculera des stats initiales basées sur l'état ACTUEL, ce qui est faux)
        army = cls(army_id, units, general)
        
        # Restaurer les vraies stats initiales
        stats = data.get('initial_stats', {})
        if stats:
            army.initial_count = stats.get('count', army.initial_count)
            army.initial_total_hp = stats.get('hp', army.initial_total_hp)
            army.initial_units_breakdown = stats.get('breakdown', army.initial_units_breakdown)
            
        return army

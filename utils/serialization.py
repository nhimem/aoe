# utils/serialization.py
import json
import sys
from engine import Engine

def save_game(engine: Engine, filename: str):
    """
    Sauvegarde l'état complet du moteur de jeu dans un fichier JSON (.json).
    Remplace pickle pour sécurité (Req 12 patchée).
    """
    # Force extension .json if .sav is passed (migration)
    if filename.endswith(".sav"):
        filename = filename[:-4] + ".json"
        
    print(f"Sauvegarde de la partie dans '{filename}'...")
    try:
        data = engine.to_dict()
        with open(filename, 'w') as f:
            json.dump(data, f, indent=None) # Compact save
        print("Sauvegarde réussie !")
    except IOError as e:
        print(f"Erreur lors de la sauvegarde : {e}", file=sys.stderr)

def load_game(filename: str) -> Engine:
    """
    Charge un moteur de jeu complet depuis un fichier JSON (.json).
    """
    # Support legacy .sav extension in CLI but warn
    if filename.endswith(".sav"):
        print("Attention: Les sauvegardes .sav (pickle) ne sont plus supportées. Veuillez utiliser .json.", file=sys.stderr)
        # Try to find .json version
        json_path = filename[:-4] + ".json"
        if os.path.exists(json_path):
             filename = json_path
        else:
             print("Erreur: Impossible de charger l'ancien format .sav", file=sys.stderr)
             sys.exit(1)

    print(f"Chargement de la partie depuis '{filename}'...")
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
            
        engine = Engine.from_dict(data)
        
        print(f"Chargement reussi ! Tick actuel : {engine.turn_count}")
        return engine
    except FileNotFoundError:
        print(f"Erreur : Fichier de sauvegarde '{filename}' introuvable.", file=sys.stderr)
        sys.exit(1)
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Erreur : Le fichier de sauvegarde est corrompu ou invalide: {e}", file=sys.stderr)
        sys.exit(1)
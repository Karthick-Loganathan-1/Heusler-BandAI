import os
import re

PSEUDO_DIR = '/home/washingmachine/pseudo'
LOG_FILE = os.path.join(os.path.dirname(__file__), 'pseudo_mapping.log')

# Elements needed for Heusler alloys
REQUIRED_ELEMENTS = [
    'Co', 'Mn', 'Si', 'Fe', 'Ge', 'Al', 'Cr', 'Ti', 'Sn', 'V', 'Nb', 'Zr', 'Ni' , 'Ga', 'In'
]

# Preferred patterns (case-insensitive)
PREFERRED_PATTERNS = [
    r'^{el}[_\.].*PBE.*UPF$',
    r'^{el}[_\.].*pbe.*upf$',
    r'^{el}[_\.].*UPF$',
    r'^{el}[_\.].*upf$',
    r'^{el}.*PBE.*UPF$',
    r'^{el}.*pbe.*upf$',
    r'^{el}.*UPF$',
    r'^{el}.*upf$',
    r'^{el}.*',
]


def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')
    print(msg)

def find_best_pseudo(element, pseudo_files):
    for pattern in PREFERRED_PATTERNS:
        regex = re.compile(pattern.format(el=element), re.IGNORECASE)
        for fname in pseudo_files:
            if regex.match(fname):
                return fname
    return None

def map_pseudos(elements, pseudo_dir):
    pseudo_files = [f for f in os.listdir(pseudo_dir) if f.lower().endswith('.upf')]
    mapping = {}
    for el in elements:
        best = find_best_pseudo(el, pseudo_files)
        if best:
            mapping[el] = best
            log(f"[OK] {el}: {best}")
        else:
            mapping[el] = None
            log(f"[MISSING] {el}: No suitable pseudopotential found in {pseudo_dir}")
    return mapping

if __name__ == "__main__":
    mapping = map_pseudos(REQUIRED_ELEMENTS, PSEUDO_DIR)
    print("\nElement to Pseudopotential Mapping:")
    for el, fname in mapping.items():
        print(f"{el}: {fname if fname else 'MISSING'}") 
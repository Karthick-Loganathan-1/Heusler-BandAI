import os
import re
import subprocess
import time
from pseudo_mapper import map_pseudos, REQUIRED_ELEMENTS, PSEUDO_DIR
import numpy as np
import matplotlib.pyplot as plt
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QTextEdit, QListWidget, QLineEdit, QTableWidget, QTableWidgetItem, QFileDialog, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import glob

# structure builder 
def heusler_l21_structure(formula):
    # Parse formula, e.g., 'Co2MnSi' -> ['Co', 'Mn', 'Si']
    m = re.match(r'([A-Z][a-z]?)(\d*)([A-Z][a-z]?)(\d*)([A-Z][a-z]?)(\d*)', formula)
    if not m:
        raise ValueError(f"Invalid Heusler formula: {formula}")
    el1, n1, el2, n2, el3, n3 = m.groups()
    n1 = int(n1) if n1 else 1
    n2 = int(n2) if n2 else 1
    n3 = int(n3) if n3 else 1
    # L21 Heusler: X2YZ, Fm-3m, a ~5.65-6.1 Å (use 5.8 Å default)
    from pymatgen.core import Structure, Lattice
    a = 5.8
    lattice = Lattice.cubic(a)
    species = [el1, el1, el2, el3]
    coords = [
        [0, 0, 0],
        [0.5, 0.5, 0.5],
        [0.25, 0.25, 0.25],
        [0.75, 0.75, 0.75],
    ]
    struct = Structure(lattice, species, coords)
    return struct

# --- QE input generator logic ---
def generate_qe_inputs(formula, pseudo_map, out_dir, ecutwfc=40, kpoints=(8,8,8)):
    struct = heusler_l21_structure(formula)
    elements = list(set([str(s) for s in struct.species]))
    # Prepare atomic species and pseudo lines
    atomic_species = []
    for el in elements:
        pseudo = pseudo_map.get(el)
        if not pseudo:
            raise RuntimeError(f"No pseudopotential found for {el}")
        atomic_species.append(f"{el} 1.0 {pseudo}")  # Only filename, no path
    atomic_species_str = '\n'.join(atomic_species)
    # Atomic positions (crystal coordinates)
    atomic_positions = []
    for site in struct:
        atomic_positions.append(f"{site.species_string} {site.frac_coords[0]:.8f} {site.frac_coords[1]:.8f} {site.frac_coords[2]:.8f}")
    atomic_positions_str = '\n'.join(atomic_positions)
    # Ensure tmp and pseudo directories exist (relative paths)
    rel_tmp_dir = os.path.join(out_dir, 'tmp')
    rel_pseudo_dir = os.path.join(out_dir, 'pseudo')
    os.makedirs(rel_tmp_dir, exist_ok=True)
    os.makedirs(rel_pseudo_dir, exist_ok=True)
    # starting_magnetization for all species
    mag_lines = '\n'.join([f"  starting_magnetization({i+1}) = 0.5," for i in range(len(elements))])
    # SCF input (ibrav=2, celldm(1)=5.8, no CELL_PARAMETERS)
    scf_in = f"""
&CONTROL
  calculation = 'scf',
  prefix = '{formula}',
  outdir = './tmp',
  pseudo_dir = '/home/washingmachine/pseudo',
/
&SYSTEM
  ibrav = 2,
  celldm(1) = 5.8,
  nat = {len(struct.sites)},
  ntyp = {len(elements)},
  ecutwfc = {ecutwfc},
  nspin = 2,
  tot_magnetization = 0,
{mag_lines}
/
&ELECTRONS
  conv_thr = 1.0d-8,
/
ATOMIC_SPECIES
{atomic_species_str}
ATOMIC_POSITIONS crystal
{atomic_positions_str}
K_POINTS automatic
{' '.join(str(k) for k in kpoints)} 0 0 0
"""
    # NSCF input (copy, but calculation = 'nscf')
    nscf_in = scf_in.replace("calculation = 'scf'", "calculation = 'nscf'")
    # Write files
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'scf.in'), 'w') as f:
        f.write(scf_in)
    with open(os.path.join(out_dir, 'nscf.in'), 'w') as f:
        f.write(nscf_in)
    return scf_in, nscf_in

# --- QE runner logic ---
QE_BIN = 'pw.x'  # Assumes pw.x is in PATH
LOG_FILE = os.path.join(os.path.dirname(__file__), 'qe_run.log')
COMMON_FIXES = {
    'convergence has not been achieved': 'Increase ecutwfc or change mixing_beta',
    'Error in routine read_upf_v2 (1)': 'Check pseudopotential file and format',
    'No such file or directory': 'Check pseudo_dir and file names',
    'Error in routine electrons (1)': 'Try reducing conv_thr or changing mixing_beta',
    'Error in routine davcio (1)': 'Check disk space and outdir',
    'Error in routine fft (1)': 'Increase memory or reduce k-points',
    'Error in routine cdiaghg (1)': 'Try different diagonalization or smearing',
    'Error in routine checkallsym (1)': 'Check atomic positions and symmetry',
    'charge is wrong: smearing is needed': 'Set occupations = \'smearing\', smearing = \'mp\', degauss = 0.01 in &SYSTEM',
}
def log(msg):
    with open(LOG_FILE, 'a') as f:
        f.write(msg + '\n')
    print(msg)
def run_pw(input_file, work_dir, output_file):
    cmd = [QE_BIN, '-in', input_file]
    with open(os.path.join(work_dir, output_file), 'w') as out:
        proc = subprocess.run(cmd, cwd=work_dir, stdout=out, stderr=subprocess.STDOUT)
    return proc.returncode
def parse_qe_output(output_path):
    errors = []
    with open(output_path, 'r') as f:
        lines = f.readlines()
    for line in lines:
        for err, fix in COMMON_FIXES.items():
            if err in line:
                errors.append((err, fix, line.strip()))
        if 'error' in line.lower() or 'Error' in line:
            errors.append(('Unknown error', 'Check output', line.strip()))
    if any('convergence has not been achieved' in l for l in lines):
        errors.append(('SCF not converged', 'Increase ecutwfc or change mixing_beta', 'SCF not converged'))
    return errors
def fix_smearing_needed(input_path):
    """If 'charge is wrong: smearing is needed' is detected, add smearing keywords to &SYSTEM."""
    with open(input_path, 'r') as f:
        lines = f.readlines()
    new_lines = []
    in_system = False
    system_end = False
    for line in lines:
        if line.strip().startswith('&SYSTEM'):
            in_system = True
        if in_system and '/' in line:
            # Insert smearing settings before closing /
            new_lines.append("  occupations = 'smearing',\n")
            new_lines.append("  smearing = 'mp',\n")
            new_lines.append("  degauss = 0.01,\n")
            in_system = False
            system_end = True
        new_lines.append(line)
    if not system_end:
        # If &SYSTEM was not found or not closed, just append at the end
        new_lines.append("  occupations = 'smearing',\n  smearing = 'mp',\n  degauss = 0.01,\n")
    with open(input_path, 'w') as f:
        f.writelines(new_lines)
    log(f"[FIX] Added smearing keywords to {input_path} due to 'charge is wrong: smearing is needed' error.")
def run_scf_nscf_for_alloys(alloy_list, base_dir):
    results = {}
    for alloy in alloy_list:
        alloy_dir = os.path.join(base_dir, alloy)
        scf_in = os.path.join(alloy_dir, 'scf.in')
        nscf_in = os.path.join(alloy_dir, 'nscf.in')
        scf_out = os.path.join(alloy_dir, 'scf.out')
        nscf_out = os.path.join(alloy_dir, 'nscf.out')
        log(f"\n=== Running SCF for {alloy} ===")
        rc = run_pw('scf.in', alloy_dir, 'scf.out')
        errors = parse_qe_output(scf_out)
        # Handle smearing error
        if any('charge is wrong: smearing is needed' in e[0] or 'charge is wrong: smearing is needed' in e[2] for e in errors):
            log(f"[ERROR] Detected 'charge is wrong: smearing is needed' for {alloy}. Applying fix and rerunning.")
            fix_smearing_needed(scf_in)
            rc = run_pw('scf.in', alloy_dir, 'scf.out')
            errors = parse_qe_output(scf_out)
        if errors:
            log(f"[ERROR] SCF for {alloy}: {errors}")
            results[alloy] = {'scf': 'error', 'errors': errors}
            continue
        log(f"[OK] SCF for {alloy} completed.")
        log(f"\n=== Running NSCF for {alloy} ===")
        rc = run_pw('nscf.in', alloy_dir, 'nscf.out')
        errors = parse_qe_output(nscf_out)
        if errors:
            log(f"[ERROR] NSCF for {alloy}: {errors}")
            results[alloy] = {'nscf': 'error', 'errors': errors}
            continue
        log(f"[OK] NSCF for {alloy} completed.")
        results[alloy] = {'scf': 'ok', 'nscf': 'ok', 'errors': None}
    return results

# --- Band structure and DOS stubs ---
def generate_band_structure_inputs(formula, pseudo_map, out_dir, npoints_per_segment=20):
    # FCC high-symmetry points in fractional coordinates
    kpoints = {
        'G': [0.0, 0.0, 0.0],  # Gamma
        'X': [0.5, 0.0, 0.0],
        'W': [0.5, 0.25, 0.75],
        'K': [0.375, 0.375, 0.75],
        'L': [0.5, 0.5, 0.5],
        'U': [0.625, 0.25, 0.625],
    }
    # Path: Γ → X → W → K → Γ → L → U → W → L → K → U → X
    path = [
        ('G', 'X'), ('X', 'W'), ('W', 'K'), ('K', 'G'),
        ('G', 'L'), ('L', 'U'), ('U', 'W'), ('W', 'L'),
        ('L', 'K'), ('K', 'U'), ('U', 'X')
    ]
    # Interpolate k-points along the path
    kpt_list = []
    labels = []
    for seg in path:
        start, end = kpoints[seg[0]], kpoints[seg[1]]
        for i in range(npoints_per_segment):
            frac = i / npoints_per_segment
            kpt = [start[j] + frac * (end[j] - start[j]) for j in range(3)]
            kpt_list.append(kpt)
            labels.append(seg[0] if i == 0 else '')
    kpt_list.append(kpoints[path[-1][1]])
    labels.append(path[-1][1])
    # Write kpoints_band.in
    kpt_file = os.path.join(out_dir, 'kpoints_band.in')
    with open(kpt_file, 'w') as f:
        f.write(f"{len(kpt_list)}\n")
        for k, label in zip(kpt_list, labels):
            lab = label if label else '-'
            f.write(f"  {k[0]:.8f} {k[1]:.8f} {k[2]:.8f} {lab}\n")
    # Write bands.in (reuse SCF input, but calculation = 'bands', restart_mode = 'restart')
    bands_in = f"""
&CONTROL
  calculation = 'bands',
  prefix = '{formula}',
  outdir = './out',
  pseudo_dir = '{PSEUDO_DIR}',
  restart_mode = 'restart',
/
&SYSTEM
  ibrav = 0,
  nat = 4,
  ntyp = 3,
  ecutwfc = 40,
  nspin = 2,
/
&ELECTRONS
  conv_thr = 1.0d-8,
/
ATOMIC_SPECIES
"""
    struct = heusler_l21_structure(formula)
    elements = list(set([str(s) for s in struct.species]))
    for el in elements:
        pseudo = pseudo_map.get(el)
        bands_in += f"{el} 1.0 {os.path.join(PSEUDO_DIR, pseudo)}\n"
    cell = struct.lattice.matrix
    bands_in += "CELL_PARAMETERS angstrom\n"
    for row in cell:
        bands_in += f"{' '.join(f'{x:.8f}' for x in row)}\n"
    bands_in += "ATOMIC_POSITIONS angstrom\n"
    for site in struct:
        bands_in += f"{site.species_string} {site.a:.8f} {site.b:.8f} {site.c:.8f}\n"
    bands_in += "K_POINTS crystal_b\n"
    bands_in += f"{len(kpt_list)}\n"
    for k in kpt_list:
        bands_in += f"  {k[0]:.8f} {k[1]:.8f} {k[2]:.8f} 1\n"
    with open(os.path.join(out_dir, 'bands.in'), 'w') as f:
        f.write(bands_in)
    return kpt_file, os.path.join(out_dir, 'bands.in')

def generate_dos_inputs(formula, pseudo_map, out_dir, kpoints=(12,12,12)):
    struct = heusler_l21_structure(formula)
    elements = list(set([str(s) for s in struct.species]))
    atomic_species = []
    for el in elements:
        pseudo = pseudo_map.get(el)
        if not pseudo:
            raise RuntimeError(f"No pseudopotential found for {el}")
        atomic_species.append(f"{el} 1.0 {pseudo}")
    atomic_species_str = '\n'.join(atomic_species)
    atomic_positions = []
    for site in struct:
        atomic_positions.append(f"{site.species_string} {site.frac_coords[0]:.8f} {site.frac_coords[1]:.8f} {site.frac_coords[2]:.8f}")
    atomic_positions_str = '\n'.join(atomic_positions)
    mag_lines = '\n'.join([f"  starting_magnetization({i+1}) = 0.5," for i in range(len(elements))])
    dos_in = f"""
&CONTROL
  calculation = 'scf',
  prefix = '{formula}',
  outdir = './tmp',
  pseudo_dir = '/home/washingmachine/pseudo',
/
&SYSTEM
  ibrav = 2,
  celldm(1) = 5.8,
  nat = {len(struct.sites)},
  ntyp = {len(elements)},
  ecutwfc = 40,
  nspin = 2,
  occupations = 'smearing',
  smearing = 'mp',
  degauss = 0.01,
{mag_lines}
/
&ELECTRONS
  conv_thr = 1.0d-8,
/
ATOMIC_SPECIES
{atomic_species_str}
ATOMIC_POSITIONS crystal
{atomic_positions_str}
K_POINTS automatic
{' '.join(str(k) for k in kpoints)} 0 0 0
"""
    with open(os.path.join(out_dir, 'dos.in'), 'w') as f:
        f.write(dos_in)
    return os.path.join(out_dir, 'dos.in')

def run_band_structure(alloy, alloy_dir):
    # Run pw.x for bands
    log(f"\n=== Running Bands for {alloy} ===")
    rc = run_pw('bands.in', alloy_dir, 'bands.out')
    if rc != 0:
        log(f"[ERROR] Bands calculation failed for {alloy}")
        return False
    # Run bands.x postprocessing
    bands_dat = os.path.join(alloy_dir, 'bands.dat')
    bands_pp_in = os.path.join(alloy_dir, 'bands_pp.in')
    with open(bands_pp_in, 'w') as f:
        f.write(f"&BANDS\n  prefix = '{alloy}',\n  outdir = './out',\n  filband = '{bands_dat}'\n/\n")
    cmd = ['bands.x', '-in', 'bands_pp.in']
    with open(os.path.join(alloy_dir, 'bands_pp.out'), 'w') as out:
        subprocess.run(cmd, cwd=alloy_dir, stdout=out, stderr=subprocess.STDOUT)
    log(f"[OK] Bands calculation and postprocessing done for {alloy}")
    return True

def run_dos(alloy, alloy_dir):
    log(f"\n=== Running DOS for {alloy} ===")
    dos_in = os.path.join(alloy_dir, 'dos.in')
    dos_out = os.path.join(alloy_dir, 'dos.out')
    rc = run_pw('dos.in', alloy_dir, 'dos.out')
    if rc != 0:
        log(f"[ERROR] DOS calculation failed for {alloy}")
        return False
    # Run dos.x postprocessing
    dos_dat = os.path.join(alloy_dir, 'dos.dat')
    dos_pp_in = os.path.join(alloy_dir, 'dos_pp.in')
    with open(dos_pp_in, 'w') as f:
        f.write(f"&DOS\n  prefix = '{alloy}',\n  outdir = './tmp',\n  fildos = '{dos_dat}'\n/\n")
    cmd = ['dos.x', '-in', 'dos_pp.in']
    with open(os.path.join(alloy_dir, 'dos_pp.out'), 'w') as out:
        subprocess.run(cmd, cwd=alloy_dir, stdout=out, stderr=subprocess.STDOUT)
    if not os.path.exists(dos_dat):
        log(f"[ERROR] DOS postprocessing failed for {alloy}")
        return False
    log(f"[OK] DOS calculation and postprocessing done for {alloy}")
    return True

def plot_band_structure(alloy_dir):
    # Parse bands.dat and kpoints_band.in
    bands_dat = os.path.join(alloy_dir, 'bands.dat')
    kpt_file = os.path.join(alloy_dir, 'kpoints_band.in')
    if not os.path.exists(bands_dat) or not os.path.exists(kpt_file):
        print(f"[WARN] bands.dat or kpoints_band.in not found in {alloy_dir}")
        return
    # Parse k-point labels and positions
    k_labels = []
    k_dist = [0.0]
    prev_k = None
    with open(kpt_file) as f:
        lines = f.readlines()[1:]
        for i, line in enumerate(lines):
            parts = line.split()
            k = np.array([float(parts[0]), float(parts[1]), float(parts[2])])
            label = parts[3] if len(parts) > 3 else ''
            if i == 0:
                prev_k = k
            else:
                k_dist.append(k_dist[-1] + np.linalg.norm(k - prev_k))
                prev_k = k
            k_labels.append(label)
    # Parse bands.dat
    with open(bands_dat) as f:
        lines = f.readlines()
    n_kpts = int(lines[0].strip())
    n_bands = int(lines[1].strip())
    bands = np.zeros((n_bands, n_kpts))
    idx = 2
    for ik in range(n_kpts):
        for ib in range(n_bands):
            bands[ib, ik] = float(lines[idx].split()[1])
            idx += 1
    # Plot
    plt.figure(figsize=(8,6))
    for ib in range(n_bands):
        plt.plot(k_dist, bands[ib], color='b', lw=1)
    # Add high-symmetry labels
    ticks = []
    tick_labels = []
    for i, label in enumerate(k_labels):
        if label != '-' and label != '':
            ticks.append(k_dist[i])
            tick_labels.append(label.replace('G', r'$\Gamma$'))
    plt.xticks(ticks, tick_labels)
    plt.xlabel('k-path')
    plt.ylabel('Energy (eV)')
    plt.title('Band Structure')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(alloy_dir, 'band_structure.png'))
    plt.close()
    print(f"[OK] Band structure plot saved to {os.path.join(alloy_dir, 'band_structure.png')}")

def plot_dos(alloy_dir):
    # Parse dos.dat or similar (assume columns: E, DOS_up, DOS_down)
    dos_file = os.path.join(alloy_dir, 'dos.dat')
    if not os.path.exists(dos_file):
        print(f"[WARN] dos.dat not found in {alloy_dir}")
        return
    data = np.loadtxt(dos_file)
    E = data[:,0]
    DOS_up = data[:,1]
    DOS_down = data[:,2] if data.shape[1] > 2 else None
    plt.figure(figsize=(6,5))
    plt.plot(E, DOS_up, label='Spin up', color='r')
    if DOS_down is not None:
        plt.plot(E, -DOS_down, label='Spin down', color='b')
    plt.xlabel('Energy (eV)')
    plt.ylabel('DOS')
    plt.title('Spin-polarized DOS')
    plt.legend()
    plt.grid(True, ls='--', lw=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(alloy_dir, 'dos.png'))
    plt.close()
    print(f"[OK] DOS plot saved to {os.path.join(alloy_dir, 'dos.png')}")

class BandDOSPlot(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(figsize=(6,4))
        super().__init__(self.fig)
        self.ax = self.fig.add_subplot(111)
    def plot_image(self, img_path):
        self.fig.clf()
        import matplotlib.pyplot as plt
        img = plt.imread(img_path)
        self.ax = self.fig.add_subplot(111)
        self.ax.imshow(img)
        self.ax.axis('off')
        self.draw()

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Heusler QE Automation GUI')
        self.resize(1100, 700)
        self.pseudo_map = map_pseudos(REQUIRED_ELEMENTS, PSEUDO_DIR)
        self.alloys = [
            'Co2MnSi', 'Co2FeSi', 'Co2MnGe', 'Co2FeGe', 'Co2MnAl', 'Co2FeAl', 'Co2CrAl', 'Co2CrSi',
            'Co2TiSn', 'Co2VSn', 'Co2NbSn', 'Co2ZrAl', 'Fe2VAl', 'Fe2TiSn', 'Fe2MnSi', 'Fe2CrSi',
            'Mn2VAl', 'Ni2MnGa', 'Ni2MnIn', 'Ni2MnSn'
        ]
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../heusler_calculations'))
        self.init_ui()
    def init_ui(self):
        layout = QHBoxLayout(self)
        # Left: Alloy list and controls
        left = QVBoxLayout()
        left.addWidget(QLabel('Heusler Alloys:'))
        self.alloy_list = QListWidget()
        self.alloy_list.addItems(self.alloys)
        left.addWidget(self.alloy_list)
        left.addWidget(QLabel('ecutwfc:'))
        self.ecutwfc_box = QSpinBox()
        self.ecutwfc_box.setRange(10, 200)
        self.ecutwfc_box.setValue(40)
        left.addWidget(self.ecutwfc_box)
        left.addWidget(QLabel('k-points:'))
        self.kpoints_box = QLineEdit('8 8 8')
        left.addWidget(self.kpoints_box)
        self.gen_btn = QPushButton('Generate Inputs')
        self.run_btn = QPushButton('Run SCF/NSCF')
        self.band_btn = QPushButton('Run Bands')
        self.dos_btn = QPushButton('Run Sp-DOS')
        self.plot_btn = QPushButton('Plot Results')
        left.addWidget(self.gen_btn)
        left.addWidget(self.run_btn)
        left.addWidget(self.band_btn)
        left.addWidget(self.dos_btn)
        left.addWidget(self.plot_btn)
        layout.addLayout(left, 2)
        # Center: Log/progress
        center = QVBoxLayout()
        center.addWidget(QLabel('Log / Progress:'))
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        center.addWidget(self.log_box)
        layout.addLayout(center, 3)
        # Right: Pseudopotential mapping and plot
        right = QVBoxLayout()
        right.addWidget(QLabel('Pseudopotential Mapping:'))
        self.pseudo_table = QTableWidget(len(REQUIRED_ELEMENTS), 2)
        self.pseudo_table.setHorizontalHeaderLabels(['Element', 'Pseudopotential'])
        for i, el in enumerate(REQUIRED_ELEMENTS):
            self.pseudo_table.setItem(i, 0, QTableWidgetItem(el))
            self.pseudo_table.setItem(i, 1, QTableWidgetItem(self.pseudo_map.get(el) or 'MISSING'))
        self.pseudo_table.resizeColumnsToContents()
        right.addWidget(self.pseudo_table)
        right.addWidget(QLabel('Band Structure / DOS:'))
        self.plot_canvas = BandDOSPlot()
        right.addWidget(self.plot_canvas)
        layout.addLayout(right, 4)
        # Connect buttons
        self.gen_btn.clicked.connect(self.generate_inputs)
        self.run_btn.clicked.connect(self.run_calculations)
        self.band_btn.clicked.connect(self.run_bands)
        self.dos_btn.clicked.connect(self.run_dos_batch)
        self.plot_btn.clicked.connect(self.plot_results)
        self.alloy_list.currentTextChanged.connect(self.plot_results)
    def log(self, msg):
        self.log_box.append(msg)
        self.log_box.ensureCursorVisible()
    def generate_inputs(self):
        ecutwfc = self.ecutwfc_box.value()
        kpoints = tuple(map(int, self.kpoints_box.text().split()))
        for i in range(self.alloy_list.count()):
            alloy = self.alloy_list.item(i).text()
            out_dir = os.path.join(self.base_dir, alloy)
            generate_qe_inputs(alloy, self.pseudo_map, out_dir, ecutwfc, kpoints)
            self.log(f"[OK] Inputs generated for {alloy}")
    def run_calculations(self):
        alloys = [self.alloy_list.item(i).text() for i in range(self.alloy_list.count())]
        run_scf_nscf_for_alloys(alloys, self.base_dir)
        self.log("[OK] SCF/NSCF calculations complete.")
    def run_bands(self):
        alloys = [self.alloy_list.item(i).text() for i in range(self.alloy_list.count())]
        for alloy in alloys:
            alloy_dir = os.path.join(self.base_dir, alloy)
            try:
                generate_band_structure_inputs(alloy, self.pseudo_map, alloy_dir)
                ok = run_band_structure(alloy, alloy_dir)
                if ok:
                    self.log(f"[OK] Bands run for {alloy}")
                else:
                    self.log(f"[ERROR] Bands failed for {alloy}")
            except Exception as e:
                self.log(f"[ERROR] Bands failed for {alloy}: {e}")
            try:
                generate_dos_inputs(alloy, self.pseudo_map, alloy_dir, kpoints=(12,12,12))
                ok = run_dos(alloy, alloy_dir)
                if ok:
                    self.log(f"[OK] DOS run for {alloy}")
                else:
                    self.log(f"[ERROR] DOS failed for {alloy}")
            except Exception as e:
                self.log(f"[ERROR] DOS failed for {alloy}: {e}")
    def run_dos_batch(self):
        alloys = [self.alloy_list.item(i).text() for i in range(self.alloy_list.count())]
        for alloy in alloys:
            alloy_dir = os.path.join(self.base_dir, alloy)
            try:
                # Generate DOS input and run DOS
                generate_dos_inputs(alloy, self.pseudo_map, alloy_dir, kpoints=(12,12,12))
                ok = run_dos(alloy, alloy_dir)
                if ok:
                    self.log(f"[OK] DOS run for {alloy}")
                else:
                    self.log(f"[ERROR] DOS failed for {alloy}")
                # Generate projwfc.in, run projwfc.x, plot spin-polarized DOS
                generate_projwfc_input(alloy, './tmp', os.path.join(alloy_dir, 'projwfc.in'))
                run_projwfc(alloy_dir)
                plot_spin_polarized_dos_from_pdos_files(alloy_dir)
                self.log(f"[OK] Spin-polarized DOS plotted for {alloy}")
            except Exception as e:
                self.log(f"[ERROR] Sp-DOS failed for {alloy}: {e}")
    def plot_results(self):
        alloy = self.alloy_list.currentItem().text() if self.alloy_list.currentItem() else None
        if not alloy:
            return
        alloy_dir = os.path.join(self.base_dir, alloy)
        band_img = os.path.join(alloy_dir, 'band_structure.png')
        dos_img = os.path.join(alloy_dir, 'dos.png')
        if os.path.exists(band_img):
            self.plot_canvas.plot_image(band_img)
            self.log(f"[OK] Displayed band structure for {alloy}")
        elif os.path.exists(dos_img):
            self.plot_canvas.plot_image(dos_img)
            self.log(f"[OK] Displayed DOS for {alloy}")
        else:
            self.plot_canvas.fig.clf()
            self.plot_canvas.draw()
            self.log(f"[WARN] No plot found for {alloy}")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == '--cli':
        # CLI mode: run all calculations and plots
        pseudo_map = map_pseudos(REQUIRED_ELEMENTS, PSEUDO_DIR)
        alloys = [
            'Co2MnSi', 'Co2FeSi', 'Co2MnGe', 'Co2FeGe', 'Co2MnAl', 'Co2FeAl', 'Co2CrAl', 'Co2CrSi',
            'Co2TiSn', 'Co2VSn', 'Co2NbSn', 'Co2ZrAl', 'Fe2VAl', 'Fe2TiSn', 'Fe2MnSi', 'Fe2CrSi',
            'Mn2VAl', 'Ni2MnGa', 'Ni2MnIn', 'Ni2MnSn'
        ]
        base_dir = os.path.join(os.path.dirname(__file__), '../heusler_calculations')
        for alloy in alloys:
            out_dir = os.path.join(base_dir, alloy)
            generate_qe_inputs(alloy, pseudo_map, out_dir)
        run_scf_nscf_for_alloys(alloys, base_dir)
        for alloy in alloys:
            alloy_dir = os.path.join(base_dir, alloy)
            try:
                generate_band_structure_inputs(alloy, pseudo_map, alloy_dir)
                run_band_structure(alloy, alloy_dir)
            except Exception as e:
                print(f"[ERROR] Bands failed for {alloy}: {e}")
            try:
                generate_dos_inputs(alloy, pseudo_map, alloy_dir, kpoints=(12,12,12))
                run_dos(alloy, alloy_dir)
            except Exception as e:
                print(f"[ERROR] DOS failed for {alloy}: {e}")
            try:
                plot_band_structure(alloy_dir)
            except Exception as e:
                print(f"[ERROR] Band plot failed for {alloy}: {e}")
            try:
                plot_dos(alloy_dir)
            except Exception as e:
                print(f"[ERROR] DOS plot failed for {alloy}: {e}")
            # --- Projwfc automation and plotting ---
            try:
                # Generate projwfc.in
                generate_projwfc_input(alloy, './tmp', os.path.join(alloy_dir, 'projwfc.in'))
                # Run projwfc.x
                run_projwfc(alloy_dir)
                # Plot spin-polarized DOS from PDOS files
                plot_spin_polarized_dos_from_pdos_files(alloy_dir)
            except Exception as e:
                print(f"[ERROR] Projwfc or PDOS plot failed for {alloy}: {e}")
    else:
        # GUI mode
        app = QApplication(sys.argv)
        mw = MainWindow()
        mw.show()
        sys.exit(app.exec_()) 
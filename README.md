# Heusler Alloy Band Gap Predictor - BandAI

This project consists of two major parts:

1. **Automated Dataset Preparation** using Quantum ESPRESSO  
2. **Machine Learning Band Gap Prediction** using Random Forest Regressor

It is focused on full-Heusler alloys with the **Fm-3m (L2₁)** structure.

---

## 1. Dataset Preparation

This phase automates the process of performing electronic structure calculations for various Heusler alloys and extracting their band gaps using Quantum ESPRESSO. The automation is done through a Python-based toolkit that supports both **GUI** and **CLI** modes.

### Highlights

- Automatically generates input files for SCF, NSCF, band structure, and DOS calculations.
- Batch executes Quantum ESPRESSO workflows for a list of Heusler alloys.
- Handles common convergence and pseudopotential issues.
- Extracts spin-polarized band gaps from `bands.x` and `projwfc.x` outputs.
- Includes a PyQt5-based GUI for managing calculations and visualizations.
- CLI mode available for headless processing and large-scale batch runs.

### Modules

- **`qe_input_generator.py`**  
  Main script to drive the GUI or CLI automation. It:
  - Accepts a list of alloys.
  - Uses `pymatgen` to generate atomic structures.
  - Creates QE input files (VC-relax, SCF, NSCF, bands, DOS).
  - Manages execution of QE tools (`pw.x`, `bands.x`, `dos.x`, `projwfc.x`).
  - Handles parsing of outputs and stores data in organized folders.

- **`pseudo_mapper.py`**  
  Maps each chemical element in the alloy to its appropriate pseudopotential file. Ensures the correct UPF is used for each run.

- **`pseudo/` directory**  
  Contains pseudopotential files for all relevant elements. Must be pre-populated.

- **`heusler_calculations/`**  
  Output directory. Each alloy gets its own subdirectory containing inputs, outputs, and visualizations.

### Outputs

For each alloy (e.g. `Co₂MnSi`), the automation produces:
- Input/output files for all steps (SCF, NSCF, DOS, etc.)
- Band structure plots
- Spin-polarized DOS plots
- Extracted band gap values (used in ML)

---

## 2. Machine Learning Prediction

Using the band gaps obtained from the automated workflow, a machine learning model is trained to predict band gaps for similar full-Heusler alloys.

### Process

- A dataset of 20 predetermined Heusler compounds is used.
- Descriptors are manually or programmatically extracted (e.g., atomic number, electronegativity, VEC, etc.).
- The target variable is the DFT band gap (extracted from QE).
- A **Random Forest Regressor** is trained on this dataset.
- The trained model is evaluated using standard regression metrics (MAE, R², etc.).
- Predictions are plotted and compared with DFT values.

### Tools Used

- `pandas`, `scikit-learn`, `matplotlib` for ML and plotting
- Feature engineering from alloy composition

---

## Example Applications

- Rapid screening of novel Heusler alloys without running DFT.
- Feature importance analysis (e.g. VEC vs. band gap trends).
- Accelerating spintronic material discovery by reducing reliance on expensive calculations.

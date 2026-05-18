from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.inverse_design import search


if __name__ == "__main__":
    search({
        "delta_h_kj_mol": 0.55,
        "delta_h_peak_kj_mol": 0.12,
        "s2_j_mol_k": 150,
        "delta_s_j_mol_k": 100,
        "memory_effect_label": 1.0,
    })

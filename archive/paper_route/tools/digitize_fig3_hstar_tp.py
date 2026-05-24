from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "extracted" / "data_points"
OUT.mkdir(parents=True, exist_ok=True)


# Approximate values digitized from Fig.3b.
# H* is read from the blue-square right y-axis. Tp is a coarse read from Fig.3a
# at the Rh=1000 K/s level, included for traceability rather than as a primary
# fitted target.
ROWS = [
    {"annealing_temperature_k": 383, "annealing_time_s": 0.1, "h_star_kj_mol": 92, "tp_1000kps_k": 377},
    {"annealing_temperature_k": 383, "annealing_time_s": 0.2, "h_star_kj_mol": 92, "tp_1000kps_k": 380},
    {"annealing_temperature_k": 383, "annealing_time_s": 0.5, "h_star_kj_mol": 94, "tp_1000kps_k": 384},
    {"annealing_temperature_k": 383, "annealing_time_s": 1, "h_star_kj_mol": 98, "tp_1000kps_k": 389},
    {"annealing_temperature_k": 383, "annealing_time_s": 2, "h_star_kj_mol": 112, "tp_1000kps_k": 397},
    {"annealing_temperature_k": 383, "annealing_time_s": 5, "h_star_kj_mol": 148, "tp_1000kps_k": 408},
    {"annealing_temperature_k": 383, "annealing_time_s": 10, "h_star_kj_mol": 155, "tp_1000kps_k": 414},
    {"annealing_temperature_k": 383, "annealing_time_s": 20, "h_star_kj_mol": 156, "tp_1000kps_k": 419},
    {"annealing_temperature_k": 383, "annealing_time_s": 50, "h_star_kj_mol": 157, "tp_1000kps_k": 424},
    {"annealing_temperature_k": 383, "annealing_time_s": 100, "h_star_kj_mol": 156, "tp_1000kps_k": 427},
    {"annealing_temperature_k": 383, "annealing_time_s": 200, "h_star_kj_mol": 155, "tp_1000kps_k": 430},
    {"annealing_temperature_k": 383, "annealing_time_s": 500, "h_star_kj_mol": 155, "tp_1000kps_k": 432},
    {"annealing_temperature_k": 383, "annealing_time_s": 1000, "h_star_kj_mol": 156, "tp_1000kps_k": 433},
    {"annealing_temperature_k": 383, "annealing_time_s": 2000, "h_star_kj_mol": 157, "tp_1000kps_k": 434},
]


def main() -> Path:
    path = OUT / "fig3b_hstar_tp_digitized.csv"
    fields = [
        "annealing_temperature_k",
        "annealing_time_s",
        "h_star_kj_mol",
        "tp_1000kps_k",
        "source",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in ROWS:
            writer.writerow({**row, "source": "manual_digitization_from_fig3a_b"})
    print(path)
    return path


if __name__ == "__main__":
    main()

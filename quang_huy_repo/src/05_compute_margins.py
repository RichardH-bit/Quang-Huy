from __future__ import annotations

import logging
import math
from pathlib import Path

import pandas as pd

from utils import ensure_dir, load_config, parse_args, setup_logging


DIRECTION_MAP = {
    "S-I": "M_SI_mm",
    "A-P": "M_AP_mm",
    "L-R": "M_LR_mm",
}


def main() -> None:
    args = parse_args("Step 5: Compute individualized margins from calibrated PCA amplitudes")
    cfg = load_config(args.config)
    root_dir = Path(cfg["paths"]["root_dir"])
    results_dir = ensure_dir(root_dir / cfg["paths"]["results_dir"])
    tables_dir = ensure_dir(results_dir / "tables")
    logs_dir = ensure_dir(root_dir / cfg["paths"]["logs_dir"])
    setup_logging(logs_dir / "05_compute_margins.log")

    pca_summary = pd.read_csv(tables_dir / "pca_summary.csv")

    sigma_sys = float(cfg["margin"]["systematic_error_mm"])
    sigma_rand = float(cfg["margin"]["random_error_mm"])
    alpha = float(cfg["margin"]["alpha"])
    location_group = cfg["clinical"]["tumor_location_group"]
    t_stage_group = cfg["clinical"]["t_stage_group"]
    f_pos = float(cfg["margin"]["tumor_location_factor"][location_group])
    f_t = float(cfg["margin"]["t_stage_factor"][t_stage_group])

    directional_margins = {"M_SI_mm": None, "M_AP_mm": None, "M_LR_mm": None}
    rows = []
    for _, row in pca_summary.iterrows():
        amplitude_mm = float(row["amplitude_mm"])
        direction = str(row["dominant_direction"])
        margin_mm = math.sqrt((2.5 * sigma_sys) ** 2 + (0.7 * sigma_rand) ** 2 + (alpha * f_pos * f_t * amplitude_mm) ** 2)
        rows.append(
            {
                "component": row["component"],
                "dominant_direction": direction,
                "amplitude_mm": amplitude_mm,
                "f_pos": f_pos,
                "f_t": f_t,
                "alpha": alpha,
                "margin_mm": margin_mm,
            }
        )
        mapped_name = DIRECTION_MAP.get(direction)
        if mapped_name is not None:
            directional_margins[mapped_name] = margin_mm

    margins_df = pd.DataFrame(rows)
    margins_df.to_csv(tables_dir / "margins.csv", index=False)

    composite_margin_3d = math.sqrt(
        sum((value or 0.0) ** 2 for value in directional_margins.values())
    ) + 2.0
    composite_df = pd.DataFrame(
        [
            {
                **directional_margins,
                "composite_margin_3d_mm": composite_margin_3d,
                "tumor_location_group": location_group,
                "t_stage_group": t_stage_group,
            }
        ]
    )
    composite_df.to_csv(tables_dir / "composite_margin_summary.csv", index=False)

    logging.info("Saved individualized margins to %s", tables_dir / "margins.csv")
    logging.info("Composite 3D margin = %.3f mm", composite_margin_3d)
    logging.info("Margin computation complete")


if __name__ == "__main__":
    main()

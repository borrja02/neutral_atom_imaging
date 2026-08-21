import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Configuration metadata mapping optical parameters, layout geometries, and cost metrics
CONFIG_METADATA = {
    "ACH-1": {
        "family": "Achromat",
        "subfolder": "mounted_achromatic_doublets_Bcoated/ACH-1_100_100_100",
        "fr1_mm": 100,
        "fr2_mm": 100,
        "ftube_mm": 100,
        "m_total": 5.71,
        "track_length_cm": 51.75,
        "cost_eur": 578.85,
    },
    "ACH-2": {
        "family": "Achromat",
        "subfolder": "mounted_achromatic_doublets_Bcoated/ACH-2_100_100_150",
        "fr1_mm": 100,
        "fr2_mm": 100,
        "ftube_mm": 150,
        "m_total": 8.57,
        "track_length_cm": 56.75,
        "cost_eur": 578.85,
    },
    "ACH-3": {
        "family": "Achromat",
        "subfolder": "mounted_achromatic_doublets_Bcoated/ACH-3_150_100_100",
        "fr1_mm": 150,
        "fr2_mm": 100,
        "ftube_mm": 100,
        "m_total": 8.57,
        "track_length_cm": 61.75,
        "cost_eur": 514.16,
    },
    "ASP-1": {
        "family": "Aspheric",
        "subfolder": "aspheric_difflim_780_Bcoated/ASP-1_100_200_150",
        "fr1_mm": 100,
        "fr2_mm": 200,
        "ftube_mm": 150,
        "m_total": 4.29,
        "track_length_cm": 76.75,
        "cost_eur": 3099.28,
    },
    "ASP-2": {
        "family": "Aspheric",
        "subfolder": "aspheric_difflim_780_Bcoated/ASP-2_200_200_100",
        "fr1_mm": 200,
        "fr2_mm": 200,
        "ftube_mm": 100,
        "m_total": 5.71,
        "track_length_cm": 91.75,
        "cost_eur": 3714.38,
    },
    "ASP-3": {
        "family": "Aspheric",
        "subfolder": "aspheric_difflim_780_Bcoated/ASP-3_100_200_200",
        "fr1_mm": 100,
        "fr2_mm": 200,
        "ftube_mm": 200,
        "m_total": 5.71,
        "track_length_cm": 81.75,
        "cost_eur": 3714.38,
    },
}


def parse_huygens_psf(file_path, threshold_mode="1/e2"):
    """Parses a Huygens PSF cross-section file to extract Strehl ratio and calculate beam waist radius."""
    file_path = Path(file_path)
    # Attempt UTF-16 decoding first, fallback to UTF-8 for legacy file exports
    try:
        content = file_path.read_text(encoding="utf-16")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="utf-8")

    # Extract Strehl ratio value using regex pattern matching
    strehl_match = re.search(r"Strehl\s+ratio:\s*([\d,]+)", content)
    if not strehl_match:
        raise ValueError(f"Strehl ratio missing in {file_path.name}")

    strehl_val = float(strehl_match.group(1).replace(",", "."))

    # Evaluate intensity threshold value based on specified mode string or float
    if isinstance(threshold_mode, str):
        clean_mode = threshold_mode.strip().lower()
        if clean_mode == "1/e":
            threshold_ratio = 1.0 / np.e
        elif clean_mode == "1/e2":
            threshold_ratio = 1.0 / (np.e**2)
        elif clean_mode.endswith("%"):
            threshold_ratio = float(clean_mode[:-1].replace(",", ".")) / 100.0
        else:
            val = float(clean_mode.replace(",", "."))
            threshold_ratio = val / 100.0 if val > 1.0 else val
    else:
        threshold_ratio = (
            threshold_mode / 100.0 if threshold_mode > 1.0 else float(threshold_mode)
        )

    data_lines = []
    is_data_zone = False

    # Extract spatial coordinate positions and intensity profiles from file body
    for line in content.splitlines():
        line = line.strip()
        if "Data" in line and "Position" in line and "Value" in line:
            is_data_zone = True
            continue
        if is_data_zone and line:
            parts = line.split()
            if len(parts) >= 3 and parts[0].isdigit():
                pos = float(parts[1].replace(",", "."))
                val = float(parts[2].replace(",", "."))
                data_lines.append((pos, val))

    if not data_lines:
        raise ValueError(f"No valid PSF data rows in {file_path.name}")

    positions, values = zip(*data_lines)
    positions = np.array(positions)
    values = np.array(values)

    # Locate peak intensity and normalize profile curve
    max_idx = np.argmax(values)
    max_val = values[max_idx]

    normalized_values = values / max_val
    positive_positions = positions[max_idx:] - positions[max_idx]
    positive_values = normalized_values[max_idx:]

    # Calculate spatial waist radius via linear interpolation at intensity threshold
    below_indices = np.where(positive_values <= threshold_ratio)[0]
    if len(below_indices) == 0:
        spot_radius = positive_positions[-1]
    else:
        cross_idx = below_indices[0]
        if cross_idx == 0:
            spot_radius = positive_positions[0]
        else:
            p_low, p_high = (
                positive_positions[cross_idx - 1],
                positive_positions[cross_idx],
            )
            v_high, v_low = (
                positive_values[cross_idx - 1],
                positive_values[cross_idx],
            )
            spot_radius = p_low + (threshold_ratio - v_high) * (p_high - p_low) / (
                v_low - v_high
            )

    return {"strehl": strehl_val, "spot_radius_um": abs(spot_radius)}


def parse_zernike_coefficients(file_path, zernike_terms=(4, 5, 6, 7, 8, 9, 10, 11)):
    """Extracts wave wavefront error statistics and requested standard Zernike expansion coefficients."""
    file_path = Path(file_path)
    # Attempt UTF-16 decoding first, fallback to UTF-8 for legacy file exports
    try:
        content = file_path.read_text(encoding="utf-16")
    except UnicodeDecodeError:
        content = file_path.read_text(encoding="utf-8")

    results = {
        "rms_rays": None,
        "rms_fitted": None,
        "peak_to_valley": None,
        "zernike_coefficients": {},
    }

    target_keys = {f"Z {z}": z for z in zernike_terms}
    in_rays = False
    in_fitted = False

    # Extract RMS metrics, Peak-to-Valley, and targeted Zernike coefficients from text
    for line in content.splitlines():
        line_str = line.strip()

        if "Peak to Valley (to centroid)" in line_str:
            results["peak_to_valley"] = float(
                line_str.split(":")[1].split()[0].replace(",", ".")
            )

        if "From integration of the rays:" in line_str:
            in_rays, in_fitted = True, False
            continue
        elif "From integration of the fitted coefficients:" in line_str:
            in_rays, in_fitted = False, True
            continue
        elif "RMS fit error" in line_str:
            in_rays, in_fitted = False, False

        if in_rays and "RMS (to chief)" in line_str:
            results["rms_rays"] = float(
                line_str.split(":")[1].split()[0].replace(",", ".")
            )

        if in_fitted and "RMS (to chief)" in line_str:
            results["rms_fitted"] = float(
                line_str.split(":")[1].split()[0].replace(",", ".")
            )

        if line_str.startswith("Z "):
            parts = line_str.split(":")
            z_header = parts[0].strip().split()
            if len(z_header) >= 2:
                z_name = f"Z {z_header[1]}"
                if z_name in target_keys:
                    val_str = z_header[2].replace(",", ".")
                    results["zernike_coefficients"][f"Z_{target_keys[z_name]}"] = (
                        float(val_str)
                    )

    return results


def compile_full_fluorescence_dataset(root_dir, threshold_mode="1/e2", zernike_terms=(4, 5, 6, 7, 8, 9, 10, 11)):
    """Traverses optics results directory tree and compiles system analysis metrics into a pandas DataFrame."""
    root_path = Path(root_dir)
    records = []
    field_folders = {"Central (0.00 mm)": "central_fov", "Extreme (0.325 mm)": "extreme_fov"}

    # Process each defined optical system configuration and field of view subfolder
    for config_id, meta in CONFIG_METADATA.items():
        results_dir = root_path / meta["subfolder"] / "results"

        for field_label, folder_name in field_folders.items():
            field_dir = results_dir / folder_name
            if not field_dir.exists():
                continue

            psf_x_file = field_dir / "huygens_cross_section_x.txt"
            psf_y_file = field_dir / "huygens_cross_section_y.txt"
            zernike_file = field_dir / "zernike_coeffs.txt"

            psf_x_data = parse_huygens_psf(psf_x_file, threshold_mode=threshold_mode)
            psf_y_data = parse_huygens_psf(psf_y_file, threshold_mode=threshold_mode)
            zernike_data = parse_zernike_coefficients(
                zernike_file, zernike_terms=zernike_terms
            )

            # Consolidate metadata specs and extracted wave/PSF parameters into a flat record
            record = {
                "config_id": config_id,
                "family": meta["family"],
                "field": field_label,
                "field_type": "central" if "central" in folder_name else "extreme",
                "cost_eur": meta["cost_eur"],
                "track_length_cm": meta["track_length_cm"],
                "m_total": meta["m_total"],
                "fr1_mm": meta["fr1_mm"],
                "fr2_mm": meta["fr2_mm"],
                "ftube_mm": meta["ftube_mm"],
                "strehl": psf_x_data["strehl"],
                "spot_radius_x_um": psf_x_data["spot_radius_um"],
                "spot_radius_y_um": psf_y_data["spot_radius_um"],
                "rms_rays": zernike_data["rms_rays"],
                "rms_fitted": zernike_data["rms_fitted"],
                "peak_to_valley": zernike_data["peak_to_valley"],
            }

            for z_key, z_val in zernike_data["zernike_coefficients"].items():
                record[z_key] = z_val

            records.append(record)

    df = pd.DataFrame(records)
    return df

def plot_configuration_performance_bars(df, configs=None, text_size=8):
    """Plots Strehl Ratio and Spot Radii (X/Y for both Central and Extreme FoV) in a compact 1x2 subplot layout."""
    # Configure typography weights globally
    plt.rcParams.update(
        {
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.weight": "bold",
        }
    )

    # Theme color definitions
    main_color = "#450474"
    bg_color = "#f4f0fa"
    grid_color = "#b0a8b9"
    marechal_color = "gray"

    # Color palette for spot radii categories
    color_central_x = "#450474"
    color_central_y = "#7b2cbf"
    color_extreme_x = "mediumorchid"
    color_extreme_y = "orchid"

    # Filter input dataframe by target optical configurations if provided
    if configs is not None:
        df_sub = df[df["config_id"].isin(configs)].copy()
    else:
        df_sub = df.copy()

    config_ids = df_sub["config_id"].unique()
    num_configs = len(config_ids)

    # Initialize compact 1x2 subplot canvas
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="tight")
    x = np.arange(num_configs)

    # Subplot 0: Strehl Ratio comparative bars
    ax0 = axes[0]
    w_strehl = 0.32

    # Extract central and extreme FoV Strehl ratios per configuration
    strehl_central = [
        df_sub[
            (df_sub["config_id"] == cid) & (df_sub["field_type"] == "central")
        ]["strehl"].values[0]
        for cid in config_ids
    ]
    strehl_extreme = [
        df_sub[
            (df_sub["config_id"] == cid) & (df_sub["field_type"] == "extreme")
        ]["strehl"].values[0]
        for cid in config_ids
    ]

    bars1 = ax0.bar(
        x - w_strehl / 2,
        strehl_central,
        w_strehl,
        label="Central FoV",
        color=color_central_x,
        edgecolor="k",
        linewidth=1.2,
        zorder=4,
    )
    bars2 = ax0.bar(
        x + w_strehl / 2,
        strehl_extreme,
        w_strehl,
        label="Extreme FoV",
        color=color_extreme_x,
        edgecolor="k",
        linewidth=1.2,
        zorder=4,
    )

    # Draw Maréchal diffraction limit reference threshold
    ax0.axhline(
        0.80,
        color=marechal_color,
        linestyle="-.",
        linewidth=1.8,
        zorder=3,
        label="Maréchal limit (0.80)",
    )
    ax0.set_ylabel("Strehl ratio", fontsize=15, color=main_color)
    ax0.set_ylim(0, 1.12)

    # Annotate bar height numerical values above bars
    for b in bars1 + bars2:
        h = b.get_height()
        ax0.annotate(
            f"{h:.2f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=text_size,
            fontweight="bold",
            color=main_color,
        )

    # Subplot 1: Spot Radii (Central X/Y and Extreme X/Y)
    ax1 = axes[1]
    w_spot = 0.18

    # Extract spot radius coordinates per field type and direction
    cx = [
        df_sub[
            (df_sub["config_id"] == cid) & (df_sub["field_type"] == "central")
        ]["spot_radius_x_um"].values[0]
        for cid in config_ids
    ]
    cy = [
        df_sub[
            (df_sub["config_id"] == cid) & (df_sub["field_type"] == "central")
        ]["spot_radius_y_um"].values[0]
        for cid in config_ids
    ]
    ex = [
        df_sub[
            (df_sub["config_id"] == cid) & (df_sub["field_type"] == "extreme")
        ]["spot_radius_x_um"].values[0]
        for cid in config_ids
    ]
    ey = [
        df_sub[
            (df_sub["config_id"] == cid) & (df_sub["field_type"] == "extreme")
        ]["spot_radius_y_um"].values[0]
        for cid in config_ids
    ]

    b_cx = ax1.bar(
        x - 1.5 * w_spot,
        cx,
        w_spot,
        label=r"Central FoV ($\text{Spot}_x$)",
        color=color_central_x,
        edgecolor="k",
        linewidth=1.1,
        zorder=4,
    )
    b_cy = ax1.bar(
        x - 0.5 * w_spot,
        cy,
        w_spot,
        label=r"Central FoV ($\text{Spot}_y$)",
        color=color_central_y,
        edgecolor="k",
        linewidth=1.1,
        zorder=4,
    )
    b_ex = ax1.bar(
        x + 0.5 * w_spot,
        ex,
        w_spot,
        label=r"Extreme FoV ($\text{Spot}_x$)",
        color=color_extreme_x,
        edgecolor="k",
        linewidth=1.1,
        zorder=4,
    )
    b_ey = ax1.bar(
        x + 1.5 * w_spot,
        ey,
        w_spot,
        label=r"Extreme FoV ($\text{Spot}_y$)",
        color=color_extreme_y,
        edgecolor="k",
        linewidth=1.1,
        zorder=4,
    )

    ax1.set_ylabel(r"Spot radius [$\mu$m]", fontsize=15, color=main_color)
    max_spot = max(max(cx), max(cy), max(ex), max(ey))
    ax1.set_ylim(0, max_spot * 1.25)

    # Annotate spot size values above respective bars
    for b in b_cx + b_cy + b_ex + b_ey:
        h = b.get_height()
        ax1.annotate(
            f"{h:.2f}",
            xy=(b.get_x() + b.get_width() / 2, h),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=text_size,
            fontweight="bold",
            color=main_color,
        )

    # Format axis lines, background colors, and legend frames
    for i, ax in enumerate(axes):
        ax.set_xticks(x)
        ax.set_xticklabels(config_ids, fontsize=13, color=main_color)
        ax.set_facecolor(bg_color)
        ax.grid(True, which="both", ls="--", alpha=0.4, color=grid_color)
        ax.tick_params(
            axis="both",
            which="both",
            labelsize=13,
            labelcolor=main_color,
            color=main_color,
            width=1.5,
            length=4,
        )
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_weight("bold")
        for spine in ax.spines.values():
            spine.set_edgecolor(main_color)
            spine.set_linewidth(1.5)

        leg = ax.legend(
            fontsize=11,
            prop={"weight": "bold", "size": 11},
            facecolor=bg_color,
            edgecolor=main_color,
            loc="upper right",
        )
        leg.get_frame().set_linewidth(1.0)
        for text in leg.get_texts():
            text.set_color(main_color)

    plt.show()


def plot_zernike_coefficients(df, configs=None, threshold=1e-3, text_size=8):
    """Plots Zernike coefficient amplitudes split by FoV with panel-specific legends and shared Y-axis in a compact layout."""
    # Configure typography weights globally
    plt.rcParams.update(
        {
            "axes.labelweight": "bold",
            "axes.titleweight": "bold",
            "font.weight": "bold",
        }
    )

    # Theme color definitions
    main_color = "#450474"
    bg_color = "#f4f0fa"
    grid_color = "#b0a8b9"

    colors = [
        "#450474",
        "#7b2cbf",
        "mediumorchid",
        "orchid",
        "#00bbf9",
        "#00f5d4",
    ]

    # Map Noll standard indexing to explicit optical aberration names
    zernike_names = {
        4: "Defocus",
        5: "Astigmatism 45°",
        6: "Astigmatism 0°",
        7: "Coma Y",
        8: "Coma X",
        9: "Trefoil Y",
        10: "Trefoil X",
        11: "Spherical",
    }

    # Filter target dataset by configuration identifiers
    if configs is not None:
        df_sub = df[df["config_id"].isin(configs)].copy()
    else:
        df_sub = df.copy()

    candidate_cols = [
        c
        for c in df_sub.columns
        if "zernike" in c.lower() or c.lower().startswith("z")
    ]

    config_ids = df_sub["config_id"].unique()
    num_configs = len(config_ids)

    fig, axes = plt.subplots(
        1, 2, figsize=(8, 4), sharey=True, layout="tight"
    )
    field_data = [
        ("central", "Central FoV"),
        ("extreme", "Extreme FoV"),
    ]

    global_max = df_sub[candidate_cols].abs().max().max()

    # Iterate over central and extreme FoV panels
    for ax_idx, (ftype, panel_title) in enumerate(field_data):
        ax = axes[ax_idx]
        df_field = df_sub[df_sub["field_type"] == ftype]

        # Retain Zernike terms exceeding the significance threshold
        panel_active_cols = [
            c for c in candidate_cols if df_field[c].abs().max() > threshold
        ]

        ax.set_title(panel_title, fontsize=15, color=main_color, pad=8)

        if not panel_active_cols:
            continue

        num_terms = len(panel_active_cols)
        total_group_width = 0.65
        bar_width = total_group_width / num_terms
        x = np.arange(num_configs)

        # Plot individual Zernike term amplitude bars per configuration
        for i, col in enumerate(panel_active_cols):
            offset = (i - (num_terms - 1) / 2) * bar_width
            values = [
                df_field[df_field["config_id"] == cid][col].abs().mean()
                if not df_field[df_field["config_id"] == cid].empty
                else 0.0
                for cid in config_ids
            ]

            try:
                z_num = int("".join(filter(str.isdigit, col)))
                name = zernike_names.get(z_num, col)
                label_text = rf"$\mathbf{{Z_{{{z_num}}}}}$ ({name})"
            except ValueError:
                label_text = col.replace("_", " ")

            bars = ax.bar(
                x + offset,
                values,
                bar_width,
                label=label_text,
                color=colors[i % len(colors)],
                edgecolor="k",
                linewidth=1.0,
                zorder=4,
            )

            # Annotate non-zero amplitude values over individual bars
            for b in bars:
                h = b.get_height()
                if h > threshold:
                    ax.annotate(
                        f"{h:.3f}",
                        xy=(b.get_x() + b.get_width() / 2, h),
                        xytext=(0, 2),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=text_size,
                        fontweight="bold",
                        color=main_color,
                    )

        # Apply axis ticks and subpanel formatting
        ax.set_xticks(x)
        ax.set_xticklabels(config_ids, fontsize=13, color=main_color)
        ax.set_facecolor(bg_color)
        ax.grid(True, which="both", ls="--", alpha=0.4, color=grid_color)

        if ax_idx == 0:
            ax.set_ylabel(
                r"Zernike amplitude [$1/\lambda$]",
                fontsize=15,
                color=main_color,
            )

        ax.tick_params(
            axis="both",
            which="both",
            labelsize=13,
            labelcolor=main_color,
            color=main_color,
            width=1.2,
            length=3,
        )

        for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
            label.set_weight("bold")

        for spine in ax.spines.values():
            spine.set_edgecolor(main_color)
            spine.set_linewidth(1.2)

        leg = ax.legend(
            fontsize=11,
            prop={"weight": "bold", "size": 11},
            facecolor=bg_color,
            edgecolor=main_color,
            loc="upper right",
        )
        leg.get_frame().set_linewidth(0.8)
        for text in leg.get_texts():
            text.set_color(main_color)

    axes[0].set_ylim(0, global_max * 1.25)
    plt.show()


# Pipeline execution
root_directory = r"C:\Users\Borja\Desktop\TFM\results\zemax_simulations\full_fluorescence_system"
df_results = compile_full_fluorescence_dataset(root_directory)

plot_configuration_performance_bars(df_results, configs=["ACH-1", "ASP-1"])
plot_configuration_performance_bars(df_results, configs=["ACH-2", "ACH-3", "ASP-2", "ASP-3"], text_size=4)
plot_zernike_coefficients(df_results, configs=["ACH-2", "ACH-3", "ASP-2", "ASP-3"], text_size=5)
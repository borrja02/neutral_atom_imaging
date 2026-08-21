import matplotlib.pyplot as plt
import numpy as np

# Effective focal length of the objective lens in mm
OBJECTIVE_EFL = 17.5

# Numerical aperture of the objective lens
OBJECTIVE_NA = 0.65

# Maximum physical field of view in mm
MAX_PHYSICAL_FOV = 1.5

# Effective focal length of the tube lens in mm
TUBE_LENS_EFL = 200.0

# Distance between the objective lens and the tube lens in mm
INTER_STAGE_DISTANCE = 350.0

# Clear aperture diameter of the tube lens in mm
LENS_DIAMETER_TUBE = 50.8

# Full height of the camera sensor in mm
SENSOR_HEIGHT = 20.0


def propagate_single_lens_ray(y0, theta0, f_obj, d_inter, f_tube, r_tube):
    """Propagates a ray through a single tube lens optical system using paraxial approximation."""
    # Initialize arrays to store ray heights and angles at each optical plane
    y, theta = np.zeros(4), np.zeros(4)
    y[0], theta[0] = y0, theta0

    # Calculate refraction at objective lens and transfer to tube lens plane
    y[1], theta[1] = y[0] + theta[0] * f_obj, theta[0] - (
        y[0] + theta[0] * f_obj
    ) / f_obj
    y[2] = y[1] + theta[1] * d_inter

    # Apply tube lens refraction only if the ray falls within the clear aperture radius
    theta[2] = theta[1] - (y[2] / f_tube) if abs(y[2]) <= r_tube else theta[1]

    # Propagate the ray from the tube lens to the camera sensor plane
    y[3], theta[3] = y[2] + theta[2] * f_tube, theta[2]
    return y


def plot_single_tube_lens_vignetting():
    """Generates ray tracing visualization for single tube lens system highlighting vignetting."""
    # Define focal lengths, inter-stage distance, and apertures
    f_obj, f_tube, d_inter, d_obj, d_tube = (
        OBJECTIVE_EFL,
        TUBE_LENS_EFL,
        INTER_STAGE_DISTANCE,
        2.0 * OBJECTIVE_NA * OBJECTIVE_EFL,
        LENS_DIAMETER_TUBE,
    )
    # Calculate radius of tube lens and half-height of sensor
    r_tube, sensor_h = d_tube / 2.0, SENSOR_HEIGHT / 2.0

    # Convert z-axis plane positions from mm to cm for plotting scale
    z_planes = (
        np.array([0.0, f_obj, f_obj + d_inter, f_obj + d_inter + f_tube])
        / 10.0
    )
    # Define marginal ray angle and maximum field angle height
    theta_marginal, y_edge = np.arcsin(OBJECTIVE_NA), MAX_PHYSICAL_FOV / 2.0

    # Trace upper, lower, and chief rays for the on-axis beam bundle
    y_c_top = propagate_single_lens_ray(
        0.0, theta_marginal, f_obj, d_inter, f_tube, r_tube
    )
    y_c_bot = propagate_single_lens_ray(
        0.0, -theta_marginal, f_obj, d_inter, f_tube, r_tube
    )
    y_c_chief = propagate_single_lens_ray(
        0.0, 0.0, f_obj, d_inter, f_tube, r_tube
    )

    # Trace upper, lower, and chief rays for the off-axis beam bundle
    y_e_top = propagate_single_lens_ray(
        y_edge, theta_marginal, f_obj, d_inter, f_tube, r_tube
    )
    y_e_bot = propagate_single_lens_ray(
        y_edge, -theta_marginal, f_obj, d_inter, f_tube, r_tube
    )
    y_e_chief = propagate_single_lens_ray(
        y_edge, 0.0, f_obj, d_inter, f_tube, r_tube
    )

    # Initialize plot figure and background styling
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, facecolor="#f4f0fa")

    # Render on-axis beam region, boundaries, and central chief ray
    ax.fill_between(
        z_planes,
        y_c_top,
        y_c_bot,
        color="#912491",
        alpha=0.15,
        label=r"$\mathbf{On\text{-}axis\ bundle}$",
    )
    ax.plot(z_planes, y_c_top, color="#4A154B", lw=1.5)
    ax.plot(z_planes, y_c_bot, color="#4A154B", lw=1.5)
    ax.plot(z_planes, y_c_chief, color="#4A154B", lw=1.1, ls="--")

    # Render off-axis beam region, boundaries, and chief ray
    ax.fill_between(
        z_planes,
        y_e_top,
        y_e_bot,
        color="#00BCD4",
        alpha=0.15,
        label=r"$\mathbf{Off\text{-}axis\ bundle}$",
    )
    ax.plot(z_planes, y_e_top, color="#00838F", lw=1.5)
    ax.plot(z_planes, y_e_bot, color="#00838F", lw=1.5)
    ax.plot(z_planes, y_e_chief, color="#00838F", lw=1.1, ls="--")

    # Define parameters and text labels for lens elements
    lens_positions, lens_diameters = [z_planes[1], z_planes[2]], [
        d_obj + 4.0,
        d_tube,
    ]
    lens_labels = [
        r"$\mathbf{Obj}$" + "\n" + r"$\mathbf{f_{obj}}$",
        r"$\mathbf{Tube\ lens}$"
        + "\n"
        + r"$\mathbf{f_{tube}}$"
        + "\n"
        + r"$\mathbf{\varnothing=2.0''}$",
    ]

    # Draw vertical lens apertures and clear aperture clipping markers
    for z_pos, diam, label in zip(lens_positions, lens_diameters, lens_labels):
        ax.plot(
            [z_pos, z_pos],
            [-diam / 2.0, diam / 2.0],
            color="#212121",
            lw=3.0,
            zorder=4,
        )
        if "Tube" in label:
            ca_h = diam / 2.0 * 0.9
            ax.plot(
                [z_pos - 0.25, z_pos + 0.25],
                [ca_h, ca_h],
                color="k",
                lw=1.8,
                zorder=5,
            )
            ax.plot(
                [z_pos - 0.25, z_pos + 0.25],
                [-ca_h, -ca_h],
                color="k",
                lw=1.8,
                zorder=5,
            )
        ax.text(
            z_pos,
            diam / 2.0 + 1.2,
            label,
            ha="center",
            va="bottom",
            fontsize=14,
            color="#212121",
        )

    # Draw vertical line representing sensor position and add text annotation
    ax.plot(
        [z_planes[3], z_planes[3]],
        [-sensor_h, sensor_h],
        color="#212121",
        lw=3.0,
        zorder=4,
    )
    ax.text(
        z_planes[3] - 4.5,
        sensor_h + 1.2,
        r"$\mathbf{Camera\ sensor}$",
        ha="center",
        va="bottom",
        fontsize=14,
        color="#212121",
    )

    # Set up positioning and labels for focal and distance dimensions
    max_aperture, y_cota = max(lens_diameters), -max(lens_diameters) / 2.0 - 6.0
    cota_labels, plane_half_h = [
        r"$\mathbf{f_{obj}}$",
        r"$\mathbf{d}$",
        r"$\mathbf{f_{tube}}$",
    ], [0.0, d_obj / 2.0, d_tube / 2.0, sensor_h]

    # Draw dashed reference lines, horizontal dimension arrows, and distance labels
    for i in range(len(z_planes) - 1):
        zs, ze, hs = z_planes[i], z_planes[i + 1], plane_half_h[i]
        ax.plot(
            [zs, zs],
            [0.0 if i == 0 else -hs, y_cota - 0.5],
            color="#4A154B",
            ls=":",
            lw=1.0,
            alpha=0.6,
        )
        if i == len(z_planes) - 2:
            ax.plot(
                [ze, ze],
                [-plane_half_h[-1], y_cota - 0.5],
                color="#4A154B",
                ls=":",
                lw=1.0,
                alpha=0.6,
            )
        ax.annotate(
            "",
            xy=(zs, y_cota),
            xytext=(ze, y_cota),
            arrowprops=dict(
                arrowstyle="<->", color="#4A154B", lw=1.3, shrinkA=0, shrinkB=0
            ),
        )
        ax.text(
            (zs + ze) / 2.0,
            y_cota + 0.6,
            cota_labels[i],
            ha="center",
            va="bottom",
            fontsize=14,
            color="#4A154B",
        )

    # Set y-axis label, optical axis reference line, and tick parameters
    ax.set_ylabel(
        r"$\mathbf{Ray\ height\ (y)\ [mm]}$", fontsize=15, color="#4A154B"
    )
    ax.axhline(0, color="#b0a8b9", ls=":", alpha=0.6)
    ax.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax.tick_params(axis="y", colors="#4A154B", labelsize=13, width=1.4)

    # Customize plot border line style and color
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#4A154B")
        spine.set_linewidth(1.5)

    # Apply axis boundary limits, grid style, and bold font to y-axis tick marks
    ax.set_xlim(-1.5, z_planes[3] + 1.5)
    ax.set_ylim(-max_aperture / 2.0 - 9.0, max_aperture / 2.0 + 10.0)
    ax.grid(True, which="both", linestyle="--", alpha=0.3, color="#b0a8b9")
    for label in ax.get_yticklabels():
        label.set_weight("bold")

    # Add plot legend and adjust canvas spacing
    ax.legend(
        fontsize=14,
        facecolor="#f4f0fa",
        edgecolor="#4A154B",
        loc="upper left",
    )

    plt.tight_layout()
    plt.show()


# Focal lengths and lens clear aperture sizes for 4f relay configuration in mm
F_RELAY1, F_RELAY2, LENS_DIAMETER_TUBE, LENS_DIAMETER_RELAY = (
    100.0,
    100.0,
    50.8,
    50.8,
)


def propagate_pupil_relay_ray(y0, theta0, f_obj, f_r1, f_r2, f_tube):
    """Propagates a ray through a pupil-conjugated relay optical system."""
    # Initialize arrays for ray heights and angles across the 4f system
    y, theta = np.zeros(6), np.zeros(6)
    y[0], theta[0] = y0, theta0

    # Refraction at objective lens and propagation to relay lens 1
    y[1], theta[1] = y[0] + theta[0] * f_obj, theta[0] - (
        y[0] + theta[0] * f_obj
    ) / f_obj

    # Refraction at relay lens 1 and transfer to relay lens 2 plane
    y[2], theta[2] = y[1] + theta[1] * f_r1, theta[1] - (
        y[1] + theta[1] * f_r1
    ) / f_r1

    # Refraction at relay lens 2 and transfer to tube lens plane
    y[3], theta[3] = y[2] + theta[2] * (f_r1 + f_r2), theta[2] - (
        y[2] + theta[2] * (f_r1 + f_r2)
    ) / f_r2

    # Refraction at tube lens and transfer to sensor plane
    y[4], theta[4] = y[3] + theta[3] * f_r2, theta[3] - (
        y[3] + theta[3] * f_r2
    ) / f_tube

    # Compute ray height and angle at camera sensor plane
    y[5], theta[5] = y[4] + theta[4] * f_tube, theta[4]
    return y


def plot_pupil_conjugated_system():
    """Generates ray tracing visualization for a pupil-conjugated relay optical system."""
    # Retrieve system focal lengths and aperture diameters
    f_obj, f_r1, f_r2, f_tube = OBJECTIVE_EFL, F_RELAY1, F_RELAY2, TUBE_LENS_EFL
    d_obj, d_r1, d_r2, d_tube = (
        2.0 * OBJECTIVE_NA * OBJECTIVE_EFL,
        LENS_DIAMETER_RELAY,
        LENS_DIAMETER_RELAY,
        LENS_DIAMETER_TUBE,
    )
    sensor_h = SENSOR_HEIGHT / 2.0

    # Convert z-axis positions of each optical element from mm to cm for plotting
    z_planes = (
        np.array(
            [
                0.0,
                f_obj,
                f_obj + f_r1,
                f_obj + f_r1 + f_r1 + f_r2,
                f_obj + 2 * f_r1 + 2 * f_r2,
                f_obj + 2 * f_r1 + 2 * f_r2 + f_tube,
            ]
        )
        / 10.0
    )
    # Define marginal ray angle and maximum field angle height
    theta_marginal, y_edge = np.arcsin(OBJECTIVE_NA), MAX_PHYSICAL_FOV / 2.0

    # Trace upper, lower, and chief rays for on-axis light bundle
    y_c_top = propagate_pupil_relay_ray(
        0.0, theta_marginal, f_obj, f_r1, f_r2, f_tube
    )
    y_c_bot = propagate_pupil_relay_ray(
        0.0, -theta_marginal, f_obj, f_r1, f_r2, f_tube
    )
    y_c_chief = propagate_pupil_relay_ray(0.0, 0.0, f_obj, f_r1, f_r2, f_tube)

    # Trace upper, lower, and chief rays for off-axis light bundle
    y_e_top = propagate_pupil_relay_ray(
        y_edge, theta_marginal, f_obj, f_r1, f_r2, f_tube
    )
    y_e_bot = propagate_pupil_relay_ray(
        y_edge, -theta_marginal, f_obj, f_r1, f_r2, f_tube
    )
    y_e_chief = propagate_pupil_relay_ray(
        y_edge, 0.0, f_obj, f_r1, f_r2, f_tube
    )

    # Initialize plot canvas and axis properties
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, facecolor="#f4f0fa")

    # Fill on-axis ray bundle region and plot ray trajectories
    ax.fill_between(
        z_planes,
        y_c_top,
        y_c_bot,
        color="#912491",
        alpha=0.15,
        label=r"$\mathbf{On\text{-}axis\ bundle}$",
    )
    ax.plot(z_planes, y_c_top, color="#4A154B", lw=1.5)
    ax.plot(z_planes, y_c_bot, color="#4A154B", lw=1.5)
    ax.plot(z_planes, y_c_chief, color="#4A154B", lw=1.1, ls="--")

    # Fill off-axis ray bundle region and plot ray trajectories
    ax.fill_between(
        z_planes,
        y_e_top,
        y_e_bot,
        color="#00BCD4",
        alpha=0.15,
        label=r"$\mathbf{Off\text{-}axis\ bundle}$",
    )
    ax.plot(z_planes, y_e_top, color="#00838F", lw=1.5)
    ax.plot(z_planes, y_e_bot, color="#00838F", lw=1.5)
    ax.plot(z_planes, y_e_chief, color="#00838F", lw=1.1, ls="--")

    # Set parameters and text strings for objective, relay, and tube lenses
    lens_positions, lens_diameters = z_planes[1:5], [
        d_obj + 4.0,
        d_r1,
        d_r2,
        d_tube,
    ]
    lens_labels = [
        r"$\mathbf{Obj}$" + "\n" + r"$\mathbf{f_{obj}}$",
        r"$\mathbf{Relay\ 1}$"
        + "\n"
        + r"$\mathbf{f_{r1}}$"
        + "\n"
        + r"$\mathbf{\varnothing=2.0''}$",
        r"$\mathbf{Relay\ 2}$"
        + "\n"
        + r"$\mathbf{f_{r2}}$"
        + "\n"
        + r"$\mathbf{\varnothing=2.0''}$",
        r"$\mathbf{Tube\ lens}$"
        + "\n"
        + r"$\mathbf{f_{tube}}$"
        + "\n"
        + r"$\mathbf{\varnothing=2.0''}$",
    ]
    # Draw vertical lens apertures and clear aperture limits
    for z_pos, diam, label in zip(lens_positions, lens_diameters, lens_labels):
        ax.plot(
            [z_pos, z_pos],
            [-diam / 2.0, diam / 2.0],
            color="#212121",
            lw=3.0,
            zorder=4,
        )
        if "Tube" in label or "Relay" in label:
            ca_h = diam / 2.0 * 0.9
            ax.plot(
                [z_pos - 0.25, z_pos + 0.25],
                [ca_h, ca_h],
                color="k",
                lw=1.8,
                zorder=5,
            )
            ax.plot(
                [z_pos - 0.25, z_pos + 0.25],
                [-ca_h, -ca_h],
                color="k",
                lw=1.8,
                zorder=5,
            )
        ax.text(
            z_pos,
            diam / 2.0 + 1.2,
            label,
            ha="center",
            va="bottom",
            fontsize=14,
            color="#212121",
        )

    # Draw vertical line representing camera sensor plane and label it
    ax.plot(
        [z_planes[5], z_planes[5]],
        [-sensor_h, sensor_h],
        color="#212121",
        lw=3.0,
        zorder=4,
    )
    ax.text(
        z_planes[5] - 5,
        sensor_h + 1.2,
        r"$\mathbf{Camera\ sensor}$",
        ha="center",
        va="bottom",
        fontsize=14,
        color="#212121",
    )

    # Set up distance annotation labels and reference height values
    max_aperture, y_cota = max(lens_diameters), -max(lens_diameters) / 2.0 - 6.0
    cota_labels = [
        r"$\mathbf{f_{obj}}$",
        r"$\mathbf{f_{r1}}$",
        r"$\mathbf{f_{r1}+f_{r2}}$",
        r"$\mathbf{f_{r2}}$",
        r"$\mathbf{f_{tube}}$",
    ]
    plane_half_h = [
        0.0,
        d_obj / 2.0,
        d_r1 / 2.0,
        d_r2 / 2.0,
        d_tube / 2.0,
        sensor_h,
    ]

    # Render dashed extension lines, dimension arrows, and focal distance labels
    for i in range(len(z_planes) - 1):
        zs, ze, hs = z_planes[i], z_planes[i + 1], plane_half_h[i]
        ax.plot(
            [zs, zs],
            [0.0 if i == 0 else -hs, y_cota - 0.5],
            color="#4A154B",
            ls=":",
            lw=1.0,
            alpha=0.6,
        )
        if i == len(z_planes) - 2:
            ax.plot(
                [ze, ze],
                [-plane_half_h[-1], y_cota - 0.5],
                color="#4A154B",
                ls=":",
                lw=1.0,
                alpha=0.6,
            )
        ax.annotate(
            "",
            xy=(zs, y_cota),
            xytext=(ze, y_cota),
            arrowprops=dict(
                arrowstyle="<->", color="#4A154B", lw=1.3, shrinkA=0, shrinkB=0
            ),
        )
        ax.text(
            (zs + ze) / 2.0,
            y_cota + 0.6,
            cota_labels[i],
            ha="center",
            va="bottom",
            fontsize=14,
            color="#4A154B",
        )

    # Set y-axis title, center reference axis line, and tick properties
    ax.set_ylabel(
        r"$\mathbf{Ray\ height\ (y)\ [mm]}$", fontsize=15, color="#4A154B"
    )
    ax.axhline(0, color="#b0a8b9", ls=":", alpha=0.6)
    ax.tick_params(
        axis="x", which="both", bottom=False, top=False, labelbottom=False
    )
    ax.tick_params(axis="y", colors="#4A154B", labelsize=13, width=1.4)

    # Style axes spines and borders
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor("#4A154B")
        spine.set_linewidth(1.5)

    # Configure plot axis limits, background grid line styles, and tick fonts
    ax.set_xlim(-1.5, z_planes[5] + 1.5)
    ax.set_ylim(-max_aperture / 2.0 - 9.0, max_aperture / 2.0 + 10.0)
    ax.grid(True, which="both", linestyle="--", alpha=0.3, color="#b0a8b9")
    for label in ax.get_yticklabels():
        label.set_weight("bold")

    # Render plot legend and optimize layout frame size
    ax.legend(
        fontsize=14,
        facecolor="#f4f0fa",
        edgecolor="#4A154B",
        loc="upper right",
    )

    plt.tight_layout()
    plt.show()


# Main execution block to generate both optical system plots
if __name__ == "__main__":
    plot_single_tube_lens_vignetting()
    plot_pupil_conjugated_system()
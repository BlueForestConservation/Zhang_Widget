import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import VBox
from IPython.display import display

MM_PER_FT = 304.8  # 1 ft = 304.8 mm


def zhang_et(mean_annual_precip, fraction_forest_cover):
    """
    Zhang et al. ET model.
    P in mm/yr, forest fraction in [0, 1].
    Returns ET in same units as P (mm/yr).
    """
    lht = fraction_forest_cover * (1 + 2820 / mean_annual_precip) / (
        1 + 2820 / mean_annual_precip + mean_annual_precip / 1410
    )
    rht = (1 - fraction_forest_cover) * (1 + 550 / mean_annual_precip) / (
        1 + 550 / mean_annual_precip + mean_annual_precip / 1100
    )
    return (lht + rht) * mean_annual_precip



def run_zhang_app():
    """
    Build and display the Zhang ET interactive widget.
    Precipitation input is always in mm/yr; outputs can be in mm/yr or ft/yr.
    """

    # --- Units toggle (for OUTPUTS only) ---
    units_widget = widgets.ToggleButtons(
        options=[("SI (mm/yr)", "si"), ("US (ft/yr)", "us")],
        value="si",
        description="Output units:",
    )

    # --- Precip slider (always mm/yr) ---
    precip_widget = widgets.FloatSlider(
        value=1962.0,  # Yuba II value in mm
        min=100.0,
        max=3000.0,
        step=10.0,
        description="P (mm/yr):",   # fixed in mm
        continuous_update=False,
        readout_format=".1f",
    )

    # --- Forest cover sliders ---
    forest_before_widget = widgets.FloatSlider(
        value=0.8,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Forest before:",
        continuous_update=False,
        readout_format=".2f",
    )

    forest_after_widget = widgets.FloatSlider(
        value=0.4,
        min=0.0,
        max=1.0,
        step=0.01,
        description="Forest after:",
        continuous_update=False,
        readout_format=".2f",
    )

    table_output = widgets.Output()
    plot_output = widgets.Output()

    # --- Main update function ---
    def update_calculation(*args):
        units = units_widget.value  # "si" or "us"
        unit_label = "mm/yr" if units == "si" else "ft/yr"

        # Model always runs in mm
        P_mm = precip_widget.value
        f_before = forest_before_widget.value
        f_after = forest_after_widget.value

        with table_output:
            table_output.clear_output()

            if P_mm <= 0:
                print("Mean annual precipitation must be > 0.")
                return

            et_before_mm = zhang_et(P_mm, f_before)
            et_after_mm = zhang_et(P_mm, f_after)

            delta_et_mm = et_after_mm - et_before_mm
            wy_before_mm = P_mm - et_before_mm
            wy_after_mm = P_mm - et_after_mm
            delta_wy_mm = wy_after_mm - wy_before_mm

            # Convert values for display if needed
            if units == "si":
                P_disp = P_mm
                et_before_disp = et_before_mm
                et_after_disp = et_after_mm
                delta_et_disp = delta_et_mm
                wy_before_disp = wy_before_mm
                wy_after_disp = wy_after_mm
                delta_wy_disp = delta_wy_mm
            else:
                P_disp = P_mm / MM_PER_FT
                et_before_disp = et_before_mm / MM_PER_FT
                et_after_disp = et_after_mm / MM_PER_FT
                delta_et_disp = delta_et_mm / MM_PER_FT
                wy_before_disp = wy_before_mm / MM_PER_FT
                wy_after_disp = wy_after_mm / MM_PER_FT
                delta_wy_disp = delta_wy_mm / MM_PER_FT

            df = pd.DataFrame(
                {
                    "Value": [
                        P_disp,
                        f_before,
                        f_after,
                        et_before_disp,
                        et_after_disp,
                        delta_et_disp,
                        wy_before_disp,
                        wy_after_disp,
                        delta_wy_disp,
                    ]
                },
                index=[
                    f"Precipitation ({unit_label})",
                    "Forest fraction (before)",
                    "Forest fraction (after)",
                    f"ET before ({unit_label})",
                    f"ET after ({unit_label})",
                    f"ΔET (after - before, {unit_label})",
                    f"Water yield before ({unit_label})",
                    f"Water yield after ({unit_label})",
                    f"ΔWater yield (after - before, {unit_label})",
                ],
            )

            display(df)

        # --- Plot: Δforest cover vs ΔET for full range ---
        with plot_output:
            plot_output.clear_output()

            fa_vals = np.linspace(0.0, 1.0, 201)
            et_before_line_mm = zhang_et(P_mm, f_before)
            et_vals_mm = zhang_et(P_mm, fa_vals)

            delta_f = fa_vals - f_before
            delta_et_line_mm = et_vals_mm - et_before_line_mm

            delta_f_current = f_after - f_before
            delta_et_current_mm = zhang_et(P_mm, f_after) - et_before_line_mm

            # Convert ΔET to display units if needed
            if units == "si":
                delta_et_line_disp = delta_et_line_mm
                delta_et_current_disp = delta_et_current_mm
            else:
                delta_et_line_disp = delta_et_line_mm / MM_PER_FT
                delta_et_current_disp = delta_et_current_mm / MM_PER_FT

            fig, ax = plt.subplots()

            ax.plot(delta_f, delta_et_line_disp, label="ΔET vs Δforest")
            ax.scatter(
                [delta_f_current],
                [delta_et_current_disp],
                marker="*",
                s=150,
                label="Current combo",
            )

            # Black dashed crosshairs
            ax.axhline(0, linewidth=0.8, color="black", linestyle="--")
            ax.axvline(0, linewidth=0.8, color="black", linestyle="--")

            ax.set_xlabel("ΔForest cover (after - before)")
            ax.set_ylabel(f"ΔET (after - before, {unit_label})")
            ax.set_title("Change in ET vs Change in Forest Cover (Zhang et al.)")
            ax.legend()

            plt.show()

    # --- Wire callbacks ---
    units_widget.observe(update_calculation, names="value")
    for w in [precip_widget, forest_before_widget, forest_after_widget]:
        w.observe(update_calculation, names="value")

    # Initial draw
    update_calculation()

    controls = VBox(
        [
            units_widget,
            precip_widget,
            forest_before_widget,
            forest_after_widget,
        ]
    )
    ui = VBox([controls, table_output, plot_output])

    return ui

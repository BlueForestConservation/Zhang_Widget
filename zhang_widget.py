import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import VBox
from IPython.display import display


def zhang_et(mean_annual_precip, fraction_forest_cover):
    """
    Zhang et al. ET model.
    P in mm/yr, forest fraction in [0, 1].
    Returns ET in same units as P.
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
    Returns the top-level widget container, but also displays it.
    """

    # --- Widgets ---
    precip_widget = widgets.FloatSlider(
        value=1962, #~ PRISM normal for North Yuba Watershed
        min=100.0,
        max=3000.0,
        step=10.0,
        description="P (mm/yr):",
        continuous_update=False,
        readout_format=".0f",
    )

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

    def update_calculation(*args):
        P = precip_widget.value
        f_before = forest_before_widget.value
        f_after = forest_after_widget.value

        with table_output:
            table_output.clear_output()

            if P <= 0:
                print("Mean annual precipitation must be > 0.")
                return

            et_before = zhang_et(P, f_before)
            et_after = zhang_et(P, f_after)

            delta_et = et_after - et_before  # >0 means ET increased
            wy_before = P - et_before
            wy_after = P - et_after
            delta_wy = wy_after - wy_before  # >0 means WY increased

            df = pd.DataFrame(
                {
                    "Value": [
                        P,
                        f_before,
                        f_after,
                        et_before,
                        et_after,
                        delta_et,
                        wy_before,
                        wy_after,
                        delta_wy,
                    ]
                },
                index=[
                    "Precipitation (mm/yr)",
                    "Forest fraction (before)",
                    "Forest fraction (after)",
                    "ET before (mm/yr)",
                    "ET after (mm/yr)",
                    "ΔET (after - before, mm/yr)",
                    "Water yield before (mm/yr)",
                    "Water yield after (mm/yr)",
                    "ΔWater yield (after - before, mm/yr)",
                ],
            )

            display(df)

        # --- Plot: Δforest cover vs ΔET for full range ---
        with plot_output:
            plot_output.clear_output()

            fa_vals = np.linspace(0.0, 1.0, 201)
            et_before_line = zhang_et(P, f_before)
            et_vals = zhang_et(P, fa_vals)

            delta_f = fa_vals - f_before
            delta_et_line = et_vals - et_before_line

            delta_f_current = f_after - f_before
            delta_et_current = zhang_et(P, f_after) - et_before_line

            fig, ax = plt.subplots()

            ax.plot(delta_f, delta_et_line, label="ΔET vs Δforest")
            ax.scatter(
                [delta_f_current],
                [delta_et_current],
                marker="*",
                s=150,
                label="Current combo",
            )

            # Black dashed crosshairs
            ax.axhline(0, linewidth=0.8, color="black", linestyle="--")
            ax.axvline(0, linewidth=0.8, color="black", linestyle="--")

            ax.set_xlabel("ΔForest cover (after - before)")
            ax.set_ylabel("ΔET (after - before, mm/yr)")
            ax.set_title("Change in ET vs Change in Forest Cover (Zhang et al.)")
            ax.legend()

            plt.show()

    # Wire up callbacks
    for w in [precip_widget, forest_before_widget, forest_after_widget]:
        w.observe(update_calculation, names="value")

    # Initial draw
    update_calculation()

    controls = VBox(
        [
            precip_widget,
            forest_before_widget,
            forest_after_widget,
        ]
    )
    ui = VBox([controls, table_output, plot_output])

    return ui

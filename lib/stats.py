import os
import io
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
from datetime import datetime

from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image, Table, TableStyle
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image

import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler

class CalibrationStats:

    def __init__(self, cfg):
        self.cfg = cfg
        self.data = {}

    def collect_from_model(self, model, solving_time=None):

        self.data["n_correspondences"] = getattr(model, "n", None)

        self.data["file_paths"] = self.cfg["file_paths"]
        self.data["sampling_strategy"] = self.cfg["sampling"]["strategy"] if "sampling" in self.cfg else None
        self.data["sampling_value"] = self.cfg["sampling"]["value"] if "sampling" in self.cfg else None

        self.data["theta_deg"] = (
            np.degrees(model.theta.flatten())
            if hasattr(model, "theta") else np.nan
        )

        self.data["theta_reference_deg"] = (
            np.degrees(model.refBor.flatten())
            if hasattr(model, "refBor") else np.nan
        )

        self.data["theta_init_deg"] = (
            np.degrees(model.initBor.flatten())
            if hasattr(model, "initBor") else np.nan
        )

        self.data["std_theta"] = (
            np.degrees(model.std_theta.flatten())
            if hasattr(model, "std_theta") else np.nan
        )

        self.data["Cov_theta"] = getattr(model, "Cov_theta", np.nan)
        self.data["sigma0"] = getattr(model, "sigma0", np.nan)
        self.data["observability"] = getattr(model, "observability", np.nan)

        self.data["initResiduals"] = getattr(
            model, "initResiduals", np.nan
        )

        self.data["adjustedResiduals"] = getattr(
            model, "adjustedResiduals", np.nan
        )

        self.data["n_removed"] = getattr(model, "n_removed", np.nan)
        self.data["redundancy"] = getattr(model, "redundancy", np.nan)
        self.data["J_cond"] = getattr(model, "J_cond", np.nan)
        self.data["J_obs"] = getattr(model, "J_obs", np.nan)
        self.data["cost_ratio"] = self.data["J_cond"] / self.data["J_obs"] if self.data["J_obs"] not in [0, np.nan] else np.nan

        self.data["thr"] = getattr(model, "thr", np.nan)

        self.data["solving_time"] = solving_time

    def plot_residuals(self):

        initResiduals = self.data["initResiduals"]
        adjustedResiduals = self.data["adjustedResiduals"]
        thr = self.data["thr"]

        if isinstance(initResiduals, float):
            return

        fig = plt.figure()
        
        bins = np.linspace(0, np.max(initResiduals)*1, 75)

        plt.hist(initResiduals, bins=bins, alpha=0.5,
                label='Initial', density=False)

        plt.hist(adjustedResiduals, bins=bins, alpha=0.6,
                label='Adjusted', density=False)

        if thr is not None:
            plt.axvline(
                thr,
                color='r',
                linestyle='--',
                label=f'Marginalisation thr. = {thr:.3f} m'
            )

        plt.xlabel('Residual norm (m)')
        plt.ylabel('Count')
        plt.title('Residual distributions: initial, adjusted, marginalised')
        plt.legend()
        plt.grid(True)

        if self.cfg["output"].get("fig_svg", False):
            svg_path = os.path.join(
                self.cfg["output"]["folder"],
                f"{self.cfg['prj_name']}_residuals.svg"
            )

            fig.savefig(svg_path, bbox_inches="tight")
            
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        self.img_residuals = Image(buf, width=10*cm, height=7.5*cm)
        plt.close(fig)

    def plot_boresight_difference(self):

        inch_to_cm = 1/2.54

        plt.rcParams.update({
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8
        })

        fig, ax = plt.subplots(figsize=(8.5*inch_to_cm, 1.65*inch_to_cm))

        ax.set_title("Boresight angle differences with reference", pad=6)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        max_range = 0.2
        cfg = self.cfg

        theta = self.data["theta_deg"]
        std_theta = self.data["std_theta"]
        sigmas = self.cfg["sigmas"]

        if isinstance(theta, float):
            return

        refAngles = np.array(cfg['refBor']).flatten()        # deg
        estAngles = theta                                   # deg
        diffAngles = estAngles - refAngles                  # deg

        sigma_rp = sigmas['rp'] * 180 / np.pi
        sigma_y = sigmas['y'] * 180 / np.pi

        labels = ["Roll", "Pitch", "Yaw"]
        y_pos = np.arange(3)[::-1]

        plt.rcParams['axes.spines.left'] = False
        plt.rcParams['axes.spines.bottom'] = False

        ax.axvline(
            0.0,
            color="k",
            linestyle="-",
            linewidth=0.25,
            alpha=0.7,
            zorder=1
        )

        for i in range(3):

            ax.plot(
                diffAngles[i],
                y_pos[i],
                marker=".",
                markersize=8,
                color=bor_paper_colors[0],
                zorder=6,
                label="Proposed approach" if i == 0 else ""
            )

            ax.errorbar(
                diffAngles[i],
                y_pos[i],
                xerr=3.0 * std_theta[i],
                fmt="none",
                ecolor=bor_paper_colors[0],
                elinewidth=1.,
                capsize=3,
                zorder=6
            )

            if "riprocess" in cfg:

                rip_est = np.array(cfg["riprocess"]["rpy"])
                rip_std = np.array(cfg["riprocess"]["std"])
                rip_diff = rip_est - refAngles

                ax.plot(
                    rip_diff[i],
                    y_pos[i],
                    marker="s",
                    markersize=4.5,
                    markerfacecolor=bor_paper_colors[1],
                    markeredgewidth=0.,
                    zorder=5,
                    label="Plane-based" if i == 0 else ""
                )

                ax.errorbar(
                    rip_diff[i],
                    y_pos[i],
                    xerr=3.0 * rip_std[i],
                    fmt="none",
                    ecolor=bor_paper_colors[1],
                    elinewidth=0.75,
                    capsize=3,
                    zorder=5
                )

            if "dn" in cfg:

                dn_est = np.array(cfg["dn"]["rpy"])
                dn_std = np.array(cfg["dn"]["std"])
                dn_diff = dn_est - refAngles

                ax.plot(
                    dn_diff[i],
                    y_pos[i],
                    marker="D",
                    markersize=4.5,
                    markerfacecolor=bor_paper_colors[2],
                    markeredgewidth=0.,
                    zorder=5,
                    label="DN solution" if i == 0 else ""
                )

                ax.errorbar(
                    dn_diff[i],
                    y_pos[i],
                    xerr=3.0 * dn_std[i],
                    fmt="none",
                    ecolor=bor_paper_colors[2],
                    elinewidth=0.75,
                    capsize=3,
                    zorder=5,
    )



            if i == 2:
                ax.vlines(
                    [-sigma_y, sigma_y],
                    y_pos[i]-0.5,
                    y_pos[i]+0.5,
                    colors="k",
                    linestyles="dashed",
                    linewidth=1.,
                    zorder=4,
                )
                ax.plot(
                    [-sigma_y, -sigma_y, sigma_y, sigma_y],
                    [y_pos[i]-0.5, y_pos[i]+0.5,
                    y_pos[i]-0.5, y_pos[i]+0.5],
                    marker="o",
                    markersize=2,
                    color="k",
                    linestyle="None",
                    zorder=4,
                )

            else:
                ax.vlines(
                    [-sigma_rp, sigma_rp],
                    y_pos[i]-0.5,
                    y_pos[i]+0.5,
                    colors="k",
                    linestyles="dashed",
                    linewidth=1,
                    zorder=4,
                )

                ax.plot(
                    [-sigma_rp, -sigma_rp, sigma_rp, sigma_rp],
                    [y_pos[i]-0.5, y_pos[i]+0.5,
                    y_pos[i]-0.5, y_pos[i]+0.5],
                    marker="o",
                    markersize=2,
                    color="k",
                    linestyle="None",
                    zorder=4,
                )

        ax.set_xlim(-max_range, max_range)
        ax.set_ylim(-0.5, 2.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Δ angle (deg) - blue: proposed, brown: plane-based, red: DN")
        ax.grid(True, axis="x", linestyle="--", alpha=0.4)
        #ax.legend(loc="upper right", frameon=True, framealpha=0.9)

        fig.tight_layout()

        if self.cfg["output"].get("fig_svg", False):
            svg_path = os.path.join(
                self.cfg["output"]["folder"],
                f"{self.cfg['prj_name']}_bor_dif.svg"
            )
            fig.savefig(svg_path, bbox_inches="tight")
        
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        self.img_bor_diff = Image(buf, width=8.5*cm, height=3.2*cm)
        plt.close(fig)
    
    def plot_correlation_matrix(self):

        obs = self.data.get("observability")

        if not isinstance(obs, dict):
            return

        Corr = obs.get("corr_matrix")
        if Corr is None:
            return

        fig = plt.figure()
        
        color_ramp = [
            (0.0, "#B51F1F"),  
            (0.25, "#FF9191"),
            (0.5, "#ffffff"), 
            (0.75, "#FF9191"),
            (1.0, "#B51F1F"),  
        ]
        cmap = LinearSegmentedColormap.from_list("custom", color_ramp)
        plt.imshow(Corr, cmap=cmap, vmin=-1, vmax=1)
        plt.colorbar()
        plt.xticks([0,1,2], ["Roll","Pitch","Yaw"])
        plt.yticks([0,1,2], ["Roll","Pitch","Yaw"])
        plt.title("Parameter Correlation Matrix")

        plt.gca().spines['top'].set_visible(True)
        plt.gca().spines['right'].set_visible(True)
        plt.gca().spines['bottom'].set_visible(True)
        plt.gca().spines['left'].set_visible(True) 
        plt.gca().spines['top'].set_color('black')
        plt.gca().spines['right'].set_color('black')
        plt.gca().spines['bottom'].set_color('black')
        plt.gca().spines['left'].set_color('black')
        plt.gca().spines['top'].set_linewidth(1.5)
        plt.gca().spines['right'].set_linewidth(1.5)
        plt.gca().spines['bottom'].set_linewidth(1.5)   

        plt.tight_layout()

        if self.cfg["output"].get("fig_svg", False):
            svg_path = os.path.join(
                self.cfg["output"]["folder"],
                f"{self.cfg['prj_name']}_corr_matrix.svg"
                )

        fig.savefig(svg_path, bbox_inches="tight")


        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        self.img_corr_matrix = Image(buf, width=9.5*cm, height=7*cm)
        plt.close(fig)

    def generate_pdf_report(self, output_path):

        doc = SimpleDocTemplate(output_path)
        elements = []
        styles = getSampleStyleSheet()

        banner_path = "./media/libor.png"
        if os.path.exists(banner_path):
            elements.append(Image(banner_path, width=16*cm, height=4.5*cm))
            elements.append(Spacer(1, 0.2*cm))


        elements.append(Paragraph("<b>Calibration Report</b>", styles["Title"]))
        elements.append(Spacer(1, 0.2*cm))

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated: {timestamp}", styles["Normal"]))
        elements.append(Spacer(1, 0.2*cm))

        if "info" in self.cfg:
            elements.append(Paragraph(f"Info: {self.cfg['info']}", styles["Normal"]))

        elements.append(Paragraph("<b>Initialization </b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2*cm))

        init_data = [["Sampling strat.", "Sampling value", "Corres. num.", "Initial guess (deg)", "Initial mean residual (m)"]]
        init_data.append([
            str(self.data["sampling_strategy"]),
            str(self.data["sampling_value"]),
            str(self.data["n_correspondences"]),
            f"{self.data['theta_init_deg']}",
            f"{np.mean(self.data['initResiduals']):.2f}"
        ])

        self._add_table(elements, init_data, header=True)

        elements.append(Paragraph("<b>Final Solution</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2*cm))

        theta = self.data["theta_deg"]
        theta_ref = self.data["theta_reference_deg"]

        std_theta = self.data["std_theta"] 

        if theta is not None:
            sol_data = [
                ["Parameter", "Reference (deg)", "Estimated (deg)", "Difference (deg)","Posteriori sigmas (deg)"],
            ]

            for i, name in enumerate(["Roll", "Pitch", "Yaw"]):
                ref = theta_ref[i] if theta_ref is not None else 0.0
                est = theta[i]
                diff = est - ref
                sol_data.append([
                    name,
                    f"{ref:.3f}",
                    f"{est:.3f}",
                    f"{diff:.3f}",
                    f"{std_theta[i]:.4f}" 
                ])

            self._add_table(elements, sol_data, header=True)

        elements.append(Paragraph("<b>A-Posteriori Statistics</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2*cm))
        ap_data = [["σ0", "Redundancy", "Cost ratio", "Solving time (s)", "Condition number", "Final mean residual (m)"]]
        ap_data.append([
            f"{self.data['sigma0']:.2f}",
            str(self.data["redundancy"]),
            f"{self.data['cost_ratio']:.2f}",
            f"{self.data['solving_time']:.2f}",
            f"{int(self.data['observability']['cond_number'])}",
            f"{np.mean(self.data['adjustedResiduals']):.2f}"
        ])
        self._add_table(elements, ap_data, header=True)

        elements.append(Paragraph("<b>Comparisons</b>", styles["Heading2"]))
        elements.append(Spacer(1, 0.2*cm))

        if "riprocess" in self.cfg and "dn" in self.cfg:

            comp_data = [[
                "Parameter",
                "Riprocess est. (°)",
                "Riprocess sigma (°)",
                "DN est. (°)",
                "DN sigma (°)"
            ]]

            rip = self.cfg["riprocess"]
            dn  = self.cfg["dn"]

            for i, name in enumerate(["Roll", "Pitch", "Yaw"]):

                comp_data.append([
                    name,
                    f"{rip['rpy'][i]:.3f}",
                    f"{rip['std'][i]:.3f}",
                    f"{dn['rpy'][i]:.3f}",
                    f"{dn['std'][i]:.3f}",
                ])

            self._add_table(elements, comp_data, header=True)

        if "riprocess" in self.cfg:

            rip = self.cfg["riprocess"]

            plane_data = [[
                "Plane number",
                "Init. plane res. (m)",
                "Adj. plane res. (m)"
            ]]

            plane_data.append([
                str(rip.get("plane_num", "N/A")),
                f"{rip.get('p_sig_prio', np.nan):.3f}",
                f"{rip.get('p_sig_post', np.nan):.3f}",
            ])

            self._add_table(elements, plane_data, header=True)
        elements.append(Paragraph("<b>Figures </b>", styles["Heading2"]))
        elements.append(self.img_residuals)
        elements.append(Spacer(1, 0.1*cm))
        elements.append(self.img_bor_diff)
        elements.append(Spacer(1, 0.1*cm))
        elements.append(self.img_corr_matrix)

        elements.append(Paragraph("<b>Miscallenous</b>", styles["Heading2"]))
        for i,path in enumerate(self.data["file_paths"]):
            elements.append(Paragraph(f"File {i+1}: {path}", styles["Normal"]))
        doc.build(elements)

    def _add_table(self, elements, data, header=False):

        table = Table(data, hAlign='LEFT')
        style = [
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ]

        if header:
            style.append(("BACKGROUND", (0,0), (-1,0), epfl_colors[0]))  # Léman
            style.append(("TEXTCOLOR", (0,0), (-1,0), colors.white))

        table.setStyle(TableStyle(style))

        elements.append(table)
        elements.append(Spacer(1, 0.6*cm))


epfl_colors = [
    "#007480",  # Canard
    "#B51F1F",  # Groseille
    "#413D3A",  # Ardoise
    "#00A79F",  # Léman
    "#FF0000",  # Rouge
    "#CAC7C7",  # Perle
]

bor_paper_colors = [
    "#3eb1c2",  
    "#a1866fff",  
    "#e14e4e",  
]

mpl.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['axes.prop_cycle'] = cycler(color=epfl_colors)

plt.rcParams.update({
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'grid.color': '#CCCCCC',
    'grid.linestyle': '--',
    'grid.linewidth': 0.5,
    'axes.grid': True,
    'font.size': 14,
    'font.family':  ('cmr10', 'STIXGeneral'),
    'lines.linewidth': 0.75,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
})
np.set_printoptions(precision=5, suppress=True)
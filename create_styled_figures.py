from pathlib import Path
import re
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageOps, ImageDraw
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --------------------------------------------------
# Configuration
# --------------------------------------------------
ROOT = Path(".")
OUT = ROOT / "reports/styled_figures"
D3 = OUT / "3d_heatmaps"
D2 = OUT / "2d_comparisons"
DASH = OUT / "dashboard"

for p in [D3, D2, DASH]:
    p.mkdir(parents=True, exist_ok=True)

DARK = "#331621"
PINK = "#CC085E"
ORANGE = "#DE7D48"
CREAM = "#EECA98"

PALETTE = [DARK, PINK, ORANGE, CREAM]
CMAP = LinearSegmentedColormap.from_list("urbanxai", PALETTE)

plt.rcParams.update({
    "figure.facecolor": CREAM,
    "axes.facecolor": "#FFF8EF",
    "axes.edgecolor": DARK,
    "axes.labelcolor": DARK,
    "text.color": DARK,
    "xtick.color": DARK,
    "ytick.color": DARK,
    "font.family": "DejaVu Sans",
    "axes.titleweight": "bold",
    "axes.titlesize": 14,
    "savefig.dpi": 220,
})

def save_plot(fig, name, folder=D2):
    fig.savefig(folder / name, bbox_inches="tight", facecolor=CREAM)
    plt.close(fig)

print("Creating UrbanXAI styled figures...")

# --------------------------------------------------
# 1. 3D Grad-CAM and occlusion heatmaps
# --------------------------------------------------
for method in ["gradcam", "occlusion"]:
    source = ROOT / "explainability" / method
    files = sorted(source.glob("*.npy"))

    for file in files:
        data = np.load(file).astype(np.float32)
        data = np.nan_to_num(data)
        if data.ndim > 2:
            data = np.squeeze(data)
        if data.ndim != 2:
            print("Skipping unexpected array:", file)
            continue

        h, w = data.shape
        x, y = np.meshgrid(np.arange(w), np.arange(h))

        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(
            x, y, data,
            cmap=CMAP,
            linewidth=0,
            antialiased=True,
            rstride=2,
            cstride=2
        )
        ax.set_title(f"{method.upper()} — {file.stem}", pad=18)
        ax.set_xlabel("Pixel X")
        ax.set_ylabel("Pixel Y")
        ax.set_zlabel("Importance")
        ax.view_init(elev=38, azim=-128)
        fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.10,
                     label="Normalized importance")
        save_plot(fig, f"{file.stem}_3d.png", D3)

        # Interactive 3D HTML surface
        interactive = go.Figure(data=[
            go.Surface(
                z=data,
                colorscale=[
                    [0.0, DARK], [0.35, PINK],
                    [0.68, ORANGE], [1.0, CREAM]
                ],
                colorbar=dict(title="Importance")
            )
        ])
        interactive.update_layout(
            title=f"{method.upper()} — {file.stem}",
            template="plotly_white",
            paper_bgcolor=CREAM,
            font=dict(color=DARK),
            scene=dict(
                xaxis_title="Pixel X",
                yaxis_title="Pixel Y",
                zaxis_title="Importance",
                camera=dict(eye=dict(x=1.55, y=1.55, z=1.0))
            ),
            margin=dict(l=0, r=0, t=55, b=0)
        )
        interactive.write_html(
            str(D3 / f"{file.stem}_interactive.html"),
            include_plotlyjs="cdn"
        )

    print(f"{method}: processed {len(files)} arrays")

# --------------------------------------------------
# 2. Test-set experiment comparison
# --------------------------------------------------
test_models = {
    "Baseline": {"IoU": 0.4536, "Dice": 0.6241},
    "Experiment 1": {"IoU": 0.6003, "Dice": 0.7502},
    "Experiment 2": {"IoU": 0.5925, "Dice": 0.7441},
    "Experiment 3": {"IoU": 0.5951, "Dice": 0.7461},
    "Experiment 4": {"IoU": 0.5948, "Dice": 0.7459},
}

fig, ax = plt.subplots(figsize=(11, 6))
names = list(test_models)
xx = np.arange(len(names))
width = 0.34

ax.bar(xx - width/2, [test_models[n]["IoU"] for n in names],
       width, label="IoU", color=PINK)
ax.bar(xx + width/2, [test_models[n]["Dice"] for n in names],
       width, label="Dice", color=ORANGE)
ax.set_xticks(xx, names, rotation=12)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Test-set segmentation performance")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save_plot(fig, "test_model_comparison.png")

# Experiment 5 and 6 are validation-only; keep separate.
val_models = {
    "ResNet50": {"IoU": 0.4139, "Dice": 0.5855},
    "ConvNeXt-Tiny": {"IoU": 0.5073, "Dice": 0.6731},
}

fig, ax = plt.subplots(figsize=(8, 5))
names = list(val_models)
xx = np.arange(len(names))
ax.bar(xx - width/2, [val_models[n]["IoU"] for n in names],
       width, label="Validation IoU", color=PINK)
ax.bar(xx + width/2, [val_models[n]["Dice"] for n in names],
       width, label="Validation Dice", color=ORANGE)
ax.set_xticks(xx, names)
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Validation performance — separate experiments")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save_plot(fig, "validation_model_comparison.png")

# Detailed Experiment 1 test metrics at threshold 0.475.
metric_names = ["IoU", "Dice", "Precision", "Recall", "F1"]
metric_values = [0.6002, 0.7502, 0.7698, 0.7315, 0.7502]

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(metric_names, metric_values,
              color=[PINK, ORANGE, CREAM, PINK, ORANGE])
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("Experiment 1 — final test metrics (threshold 0.475)")
ax.grid(axis="y", alpha=0.2)
for b, v in zip(bars, metric_values):
    ax.text(b.get_x()+b.get_width()/2, v+0.018, f"{v:.3f}",
            ha="center", fontsize=9)
save_plot(fig, "experiment1_test_metrics.png")

# --------------------------------------------------
# 3. Urban statistics
# --------------------------------------------------
csv_path = ROOT / "urban_statistics/validation_urban_statistics.csv"
df = pd.read_csv(csv_path)

fig, ax = plt.subplots(figsize=(9, 5))
labels = ["Predicted", "Ground truth"]
counts = [
    int(df["predicted_building_count"].sum()),
    int(df["ground_truth_component_count"].sum())
]
bars = ax.bar(labels, counts, color=[PINK, ORANGE], width=0.55)
ax.set_title("Detected building components — validation set")
ax.set_ylabel("Connected components")
ax.grid(axis="y", alpha=0.2)
for b, v in zip(bars, counts):
    ax.text(b.get_x()+b.get_width()/2, v+20, f"{v:,}",
            ha="center")
save_plot(fig, "building_component_comparison.png")

fig, ax = plt.subplots(figsize=(8, 5))
built = [6.3532, 6.7498]
bars = ax.bar(["Predicted", "Ground truth"], built,
              color=[PINK, ORANGE], width=0.55)
ax.set_title("Built-up area percentage — validation set")
ax.set_ylabel("Built-up area (%)")
ax.set_ylim(0, max(built)*1.3)
ax.grid(axis="y", alpha=0.2)
for b, v in zip(bars, built):
    ax.text(b.get_x()+b.get_width()/2, v+0.08, f"{v:.4f}%",
            ha="center")
save_plot(fig, "built_up_percentage.png")

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(df["building_density_per_10000_pixels"],
        bins=18, color=PINK, edgecolor=DARK, alpha=0.9)
ax.set_title("Distribution of building density")
ax.set_xlabel("Components per 10,000 pixels")
ax.set_ylabel("Number of images")
ax.grid(axis="y", alpha=0.2)
save_plot(fig, "building_density_distribution.png")

# --------------------------------------------------
# 4. UECS
# --------------------------------------------------
uecs_text = (ROOT / "explainability/uecs/uecs_results.txt").read_text()
matches = re.findall(
    r"Image\s+(\d+):\s+Correlation=([-0-9.]+),\s+UECS=([0-9.]+)",
    uecs_text
)
uecs_ids = [int(m[0]) for m in matches]
uecs_vals = [float(m[2]) for m in matches]

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar([f"{i:03d}" for i in uecs_ids], uecs_vals,
       color=[PINK if v < 0.5 else ORANGE for v in uecs_vals])
ax.axhline(0.5, color=DARK, linestyle="--",
           linewidth=1, label="0.5 reference")
ax.set_ylim(0, 1)
ax.set_xlabel("Image index")
ax.set_ylabel("UECS")
ax.set_title("Urban Explanation Consistency Score")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
save_plot(fig, "uecs_per_image.png")

# --------------------------------------------------
# 5. Available training histories
# --------------------------------------------------
for exp in [3, 4]:
    path = ROOT / f"model_results/experiment_{exp}/training_history.npz"
    if not path.exists():
        continue
    hist = np.load(path)
    fig, ax = plt.subplots(figsize=(9, 5))

    if "loss" in hist.files:
        ax.plot(hist["loss"], marker="o", color=PINK,
                label="Training loss")
    if "val_loss" in hist.files:
        ax.plot(hist["val_loss"], marker="o", color=ORANGE,
                label="Validation loss")

    ax.set_title(f"Experiment {exp} — training history")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    save_plot(fig, f"experiment_{exp}_training_history.png")

    if "accuracy" in hist.files and "val_accuracy" in hist.files:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(hist["accuracy"], marker="o", color=PINK,
                label="Training accuracy")
        ax.plot(hist["val_accuracy"], marker="o", color=ORANGE,
                label="Validation accuracy")
        ax.set_title(f"Experiment {exp} — accuracy history")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Pixel accuracy")
        ax.legend(frameon=False)
        ax.grid(alpha=0.2)
        save_plot(fig, f"experiment_{exp}_accuracy_history.png")

# --------------------------------------------------
# 6. Segmentation comparison panels
# --------------------------------------------------
visual_dir = ROOT / "model_results/visual_outputs"
panel_count = 0

for i in range(1, 11):
    names = [
        f"test_original_{i}.png",
        f"test_groundtruth_{i}.png",
        f"test_prediction_{i}.png",
        f"test_overlay_{i}.png",
    ]
    paths = [visual_dir / n for n in names]
    if not all(p.exists() for p in paths):
        continue

    images = []
    for p in paths:
        im = Image.open(p).convert("RGB")
        im.thumbnail((420, 420))
        canvas = Image.new("RGB", (420, 420), CREAM)
        canvas.paste(im, ((420-im.width)//2, (420-im.height)//2))
        images.append(canvas)

    panel = Image.new("RGB", (4*420, 470), CREAM)
    draw = ImageDraw.Draw(panel)
    labels = ["Satellite image", "Ground truth", "Prediction", "Overlay"]
    for j, im in enumerate(images):
        panel.paste(im, (j*420, 42))
        draw.text((j*420+12, 12), labels[j], fill=DARK)

    panel.save(D2 / f"segmentation_comparison_{i:02d}.png")
    panel_count += 1

# --------------------------------------------------
# 7. Summary dashboard
# --------------------------------------------------
fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=(
        "Test Dice comparison (Experiments 1–4)",
        "Validation Dice comparison (Experiments 5–6)",
        "Predicted vs ground-truth components",
        "UECS distribution"
    )
)

fig.add_trace(go.Bar(
    x=list(test_models.keys()),
    y=[v["Dice"] for v in test_models.values()],
    marker_color=PINK,
    name="Test Dice"
), row=1, col=1)

fig.add_trace(go.Bar(
    x=list(val_models.keys()),
    y=[v["Dice"] for v in val_models.values()],
    marker_color=ORANGE,
    name="Validation Dice"
), row=1, col=2)

fig.add_trace(go.Bar(
    x=["Predicted", "Ground truth"],
    y=counts,
    marker_color=[PINK, ORANGE],
    name="Components"
), row=2, col=1)

fig.add_trace(go.Histogram(
    x=uecs_vals,
    nbinsx=10,
    marker_color=ORANGE,
    name="UECS"
), row=2, col=2)

fig.update_yaxes(range=[0, 1], row=1, col=1)
fig.update_yaxes(range=[0, 1], row=1, col=2)
fig.update_layout(
    title="UrbanXAI — Explainable Urban Building Analysis",
    template="plotly_white",
    paper_bgcolor=CREAM,
    plot_bgcolor="#FFF8EF",
    font=dict(color=DARK),
    height=850,
    showlegend=False
)
fig.write_html(
    str(DASH / "urbanxai_dashboard.html"),
    include_plotlyjs=True
)

# Summary page with links to figures
html = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>UrbanXAI Dashboard</title>
<style>
body {{ background:{CREAM}; color:{DARK}; font-family:Arial,sans-serif;
       margin:0; padding:28px; }}
h1 {{ color:{DARK}; }}
.grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr));
         gap:16px; }}
.card {{ background:#FFF8EF; border-left:6px solid {PINK};
         padding:18px; border-radius:10px; }}
.value {{ font-size:28px; font-weight:bold; color:{PINK}; }}
a {{ color:{PINK}; }}
li {{ margin:8px 0; }}
</style>
</head>
<body>
<h1>UrbanXAI — Explainable Satellite-Image-Based Urban Building Analysis</h1>
<p>SpaceNet 2 Paris | EfficientNetB3 Attention U-Net | Validation and test results shown separately</p>
<div class="grid">
<div class="card"><div>Test Dice</div><div class="value">0.7502</div><small>Experiment 1, threshold 0.475</small></div>
<div class="card"><div>Test IoU</div><div class="value">0.6002</div><small>Experiment 1, threshold 0.475</small></div>
<div class="card"><div>Mean UECS</div><div class="value">0.4843</div><small>17 valid maps of 20</small></div>
<div class="card"><div>Predicted components</div><div class="value">1,848</div><small>172 validation images</small></div>
<div class="card"><div>Built-up percentage</div><div class="value">6.3532%</div><small>Validation predictions</small></div>
<div class="card"><div>Mean density</div><div class="value">6.5577</div><small>Components per 10,000 pixels</small></div>
</div>
<h2>Interactive dashboard</h2>
<iframe src="urbanxai_dashboard.html" width="100%" height="900" style="border:0"></iframe>
<h2>3D heatmaps</h2>
<ul>
<li><a href="../3d_heatmaps/">Open 3D heatmap output folder</a></li>
</ul>
<h2>2D comparisons and statistics</h2>
<ul>
{''.join(f'<li><a href="../2d_comparisons/{p.name}">{p.name}</a></li>' for p in sorted(D2.glob('*.png')))}
</ul>
<p><b>Interpretation note:</b> UECS is a preliminary spatial-consistency measure, not a validated explanation-faithfulness score. Connected components approximate building counts; density is pixel-based and is not buildings per square kilometre.</p>
</body>
</html>
"""
(DASH / "index.html").write_text(html)

# --------------------------------------------------
# Completion summary
# --------------------------------------------------
print("\nVISUALIZATION GENERATION COMPLETE")
print("3D heatmap PNGs:", len(list(D3.glob("*_3d.png"))))
print("Interactive 3D HTMLs:", len(list(D3.glob("*_interactive.html"))))
print("2D figures:", len(list(D2.glob("*.png"))))
print("Segmentation panels:", panel_count)
print("Dashboard:", DASH / "index.html")
print("Interactive dashboard:", DASH / "urbanxai_dashboard.html")

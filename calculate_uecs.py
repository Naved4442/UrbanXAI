import numpy as np
from pathlib import Path

G_DIR = Path("explainability/gradcam")
O_DIR = Path("explainability/occlusion")
OUT = Path("explainability/uecs")
OUT.mkdir(parents=True, exist_ok=True)

results = []
excluded = []

def normalize(x):
    x = np.asarray(x, dtype=np.float64).squeeze()

    if x.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape {x.shape}")

    if not np.all(np.isfinite(x)):
        raise ValueError("Map contains non-finite values")

    spread = np.ptp(x)

    if spread < 1e-12:
        return None

    return (x - x.min()) / spread

for i in range(20):
    gp = G_DIR / f"gradcam_{i:03d}.npy"
    op = O_DIR / f"occlusion_{i:03d}.npy"

    if not gp.exists() or not op.exists():
        excluded.append((i, "Missing explanation map"))
        continue

    try:
        grad = normalize(np.load(gp))
        occ = normalize(np.load(op))

        if grad is None or occ is None:
            excluded.append((i, "Constant explanation map"))
            continue

        if grad.shape != occ.shape:
            excluded.append((i, "Map shape mismatch"))
            continue

        corr = np.corrcoef(grad.ravel(), occ.ravel())[0, 1]

        if not np.isfinite(corr):
            excluded.append((i, "Undefined correlation"))
            continue

        score = float(np.clip((corr + 1) / 2, 0, 1))
        results.append((i, float(corr), score))

    except Exception as e:
        excluded.append((i, str(e)))

out_file = OUT / "uecs_results.txt"

with open(out_file, "w") as f:
    f.write("URBAN EXPLANATION CONSISTENCY SCORE\n")
    f.write("===================================\n")
    f.write("Method: normalized Pearson spatial correlation\n")
    f.write("UECS = (Pearson correlation + 1) / 2\n")
    f.write("Constant or invalid maps are excluded from the mean.\n\n")

    for i, corr, score in results:
        f.write(
            f"Image {i:03d}: Correlation={corr:.4f}, UECS={score:.4f}\n"
        )

    f.write("\nEXCLUDED IMAGES\n")
    f.write("---------------\n")

    for i, reason in excluded:
        f.write(f"Image {i:03d}: {reason}\n")

    f.write("\nSUMMARY\n")
    f.write("-------\n")
    f.write(f"Total images checked: 20\n")
    f.write(f"Valid images: {len(results)}\n")
    f.write(f"Excluded images: {len(excluded)}\n")
    f.write(f"Coverage: {len(results) / 20 * 100:.2f}%\n")

    if results:
        f.write(
            f"Mean UECS (valid images only): "
            f"{np.mean([r[2] for r in results]):.4f}\n"
        )
    else:
        f.write("Mean UECS: Not available\n")

print("Valid images:", len(results))
print("Excluded images:", len(excluded))

if results:
    print("Mean UECS:", round(float(np.mean([r[2] for r in results])), 4))

print("Saved:", out_file)

"""Generate the 560x280 competition cover image."""
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
fig = plt.figure(figsize=(5.6, 2.8), dpi=100)
ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
ax.set_xlim(0, 1); ax.set_ylim(0, 1)

# background band
ax.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="square,pad=0",
                            fc="#0b1f33", ec="none"))
ax.add_patch(FancyBboxPatch((0, 0.0), 1, 0.16, boxstyle="square,pad=0",
                            fc="#e23b3b", ec="none"))

ax.text(0.5, 0.78, "TRIAGEGEIST", ha="center", va="center",
        color="white", fontsize=26, fontweight="bold", family="DejaVu Sans")
ax.text(0.5, 0.585, "The Waiting-Room Blind Spot", ha="center", va="center",
        color="#7fd1ff", fontsize=13.5, fontstyle="italic")
ax.text(0.5, 0.40, "Auditing acuity AI  +  predicting who deteriorates",
        ha="center", va="center", color="#d7e3ee", fontsize=9.5)
ax.text(0.5, 0.30, "before they are ever seen", ha="center", va="center",
        color="#d7e3ee", fontsize=9.5)
ax.text(0.5, 0.075, "Honest evaluation · real-world NHAMCS validation · equity audit",
        ha="center", va="center", color="white", fontsize=8, fontweight="bold")

fig.savefig(ROOT / "artifacts" / "cover.png", dpi=100)
print("saved artifacts/cover.png", "(560x280)")

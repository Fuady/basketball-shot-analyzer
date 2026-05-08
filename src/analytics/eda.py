"""
eda.py
──────
Analytics: Exploratory data analysis on basketball shot dataset.
Generates plots: outcome distribution, angle boxplots, wrist speed profiles,
feature correlations, KDE curves.

Usage:
    python src/analytics/eda.py \
        --features data/processed/features/biomech_features.csv \
        --release  data/processed/features/release_features.csv \
        --output   docs/eda_report
"""
import argparse, logging, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 120, "font.size": 11})
MAKE_COLOR, MISS_COLOR = "#2ecc71", "#e74c3c"


def plot_outcome_distribution(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    counts = df["outcome"].value_counts()
    axes[0].pie([counts.get(1,0), counts.get(0,0)], labels=["Make","Miss"],
                colors=[MAKE_COLOR,MISS_COLOR], autopct="%1.1f%%",
                wedgeprops={"edgecolor":"white","linewidth":2})
    axes[0].set_title("Shot Outcome Distribution")
    by_type = df.groupby(["shot_type","outcome"]).size().unstack(fill_value=0)
    by_type.plot(kind="bar", ax=axes[1], color=[MISS_COLOR,MAKE_COLOR], edgecolor="white")
    axes[1].set_title("Outcomes by Shot Type")
    axes[1].set_xlabel("Shot Type"); axes[1].set_ylabel("Count")
    axes[1].tick_params(axis="x", rotation=15); axes[1].legend(["Miss","Make"])
    plt.suptitle("Shot Outcome Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(output_dir/"01_outcome_distribution.png"); plt.close()
    logger.info("Saved: 01_outcome_distribution.png")


def plot_release_angles(df, output_dir):
    cols = [("release_elbow_angle","Elbow Angle (°)"),("release_knee_angle","Knee Angle (°)"),
            ("release_wrist_height","Wrist Height (norm)"),("release_wrist_speed","Wrist Speed"),
            ("pre_release_knee_bend","Pre-Release Knee Bend (°)"),("release_trunk_lean","Trunk Lean (°)")]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, (col, title) in zip(axes.flatten(), cols):
        if col not in df.columns: ax.set_visible(False); continue
        sub = df[[col,"outcome"]].dropna()
        sns.boxplot(data=sub, x="outcome", y=col,
                    palette={1:MAKE_COLOR, 0:MISS_COLOR}, ax=ax, width=0.5)
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("")
        ax.set_xticklabels(["Miss","Make"])
    plt.suptitle("Key Angles at Release: Make vs Miss", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(output_dir/"02_release_angles.png"); plt.close()
    logger.info("Saved: 02_release_angles.png")


def plot_wrist_speed_profiles(df, output_dir):
    if "wrist_speed" not in df.columns: return
    fig, ax = plt.subplots(figsize=(10, 5))
    for outcome, label, color in [(1,"Make",MAKE_COLOR),(0,"Miss",MISS_COLOR)]:
        group_data = []
        for _, grp in df[df["outcome"]==outcome].groupby("video_stem"):
            ws = grp.sort_values("frame")["wrist_speed"].values
            if len(ws) > 5:
                group_data.append(np.interp(np.linspace(0,1,50), np.linspace(0,1,len(ws)), ws))
        if group_data:
            arr = np.array(group_data)
            x = np.linspace(0, 1, 50)
            ax.plot(x, arr.mean(0), color=color, linewidth=2.5, label=f"{label} (n={len(group_data)})")
            ax.fill_between(x, arr.mean(0)-arr.std(0), arr.mean(0)+arr.std(0), color=color, alpha=0.15)
    ax.axvline(0.55, color="gray", linestyle=":", alpha=0.7, label="~Release zone")
    ax.set_xlabel("Shot Phase (normalized time)"); ax.set_ylabel("Wrist Speed")
    ax.set_title("Wrist Speed Profile: Make vs Miss", fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout(); plt.savefig(output_dir/"03_wrist_speed_profiles.png"); plt.close()
    logger.info("Saved: 03_wrist_speed_profiles.png")


def plot_correlations(df, output_dir):
    numeric = [c for c in df.select_dtypes(include=[np.number]).columns
               if c not in ["release_frame","release_position_pct"]]
    if len(numeric) < 3: return
    corr = df[numeric].corr()
    if "outcome" not in corr.columns: return
    oc = corr["outcome"].drop("outcome").sort_values(key=abs, ascending=False).head(12)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = [MAKE_COLOR if v > 0 else MISS_COLOR for v in oc.values]
    axes[0].barh(oc.index, oc.values, color=colors, edgecolor="white")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_title("Feature Correlation with Outcome", fontweight="bold")
    top = oc.index[:8].tolist() + ["outcome"]
    mask = np.triu(np.ones((len(top),len(top)), dtype=bool))
    sns.heatmap(df[top].corr(), mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, ax=axes[1], linewidths=0.5)
    axes[1].set_title("Top Feature Correlation Matrix", fontweight="bold")
    plt.suptitle("Feature Correlation Analysis", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(output_dir/"04_feature_correlations.png"); plt.close()
    logger.info("Saved: 04_feature_correlations.png")


def plot_kde(df, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (col, title) in zip(axes,[("release_elbow_angle","Elbow Angle at Release"),
                                      ("pre_release_knee_bend","Pre-Release Knee Bend")]):
        if col not in df.columns: continue
        for outcome, label, color in [(1,"Make",MAKE_COLOR),(0,"Miss",MISS_COLOR)]:
            vals = df[df["outcome"]==outcome][col].dropna()
            if len(vals) > 3:
                sns.kdeplot(vals, ax=ax, color=color, label=label, fill=True, alpha=0.25, linewidth=2)
                ax.axvline(vals.mean(), color=color, linestyle="--", linewidth=1.5,
                           label=f"{label} μ={vals.mean():.1f}°")
        ax.set_title(title, fontweight="bold"); ax.set_xlabel("Angle (°)"); ax.legend()
    plt.suptitle("Angle KDE: Make vs Miss", fontsize=13, fontweight="bold")
    plt.tight_layout(); plt.savefig(output_dir/"05_angle_kde.png"); plt.close()
    logger.info("Saved: 05_angle_kde.png")


def main(args):
    output_dir = Path(args.output); output_dir.mkdir(parents=True, exist_ok=True)
    feat_path, rel_path = Path(args.features), Path(args.release)
    features_df = pd.read_csv(feat_path) if feat_path.exists() else pd.DataFrame()
    release_df  = pd.read_csv(rel_path)  if rel_path.exists()  else pd.DataFrame()
    if release_df.empty:
        logger.error("Release features not found. Run release_detector.py first."); sys.exit(1)
    plot_outcome_distribution(release_df, output_dir)
    plot_release_angles(release_df, output_dir)
    if not features_df.empty: plot_wrist_speed_profiles(features_df, output_dir)
    plot_correlations(release_df, output_dir)
    plot_kde(release_df, output_dir)
    makes = release_df["outcome"].sum()
    logger.info(f"\n✅ EDA complete → {output_dir}")
    logger.info(f"   {len(release_df)} shots | Makes: {makes} | Misses: {len(release_df)-makes}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data/processed/features/biomech_features.csv")
    parser.add_argument("--release",  default="data/processed/features/release_features.csv")
    parser.add_argument("--output",   default="docs/eda_report")
    main(parser.parse_args())

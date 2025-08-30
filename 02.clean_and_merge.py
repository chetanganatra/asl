import pandas as pd
import numpy as np
import glob
import os

# ----------------------------
# MERGE + BALANCE SCRIPT
# ----------------------------
# Features:
#  - Reads all CSV files in a folder
#  - Cleans invalid rows (NaN or all zeros in numeric cols)
#  - Preserves 'label' column
#  - Balances dataset (same samples per label)
#  - Prints debug summary
#  - Saves merged dataset
# ----------------------------

INPUT_FOLDER = ".\\gesture_csvs"
OUTPUT_FILE = "merged_gestures_balanced.csv"

all_dfs = []

for file in glob.glob(os.path.join(INPUT_FOLDER, "*.csv")):
    print(f"[INFO] Processing {file}")
    df = pd.read_csv(file)

    if "label" not in df.columns:
        print(f"[WARNING] Skipping {file} (no 'label' column found)")
        continue

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    before = len(df)

    # Drop invalid rows
    df = df.dropna(subset=numeric_cols)
    df = df[(df[numeric_cols] != 0).any(axis=1)]

    after = len(df)
    print(f"[CLEAN] {file}: {before} -> {after} valid rows")

    all_dfs.append(df)

# Merge everything
if all_dfs:
    merged = pd.concat(all_dfs, ignore_index=True)
    print(f"[MERGED] Dataset size before balancing: {len(merged)} rows")

    # ----------------------------
    # BALANCING
    # ----------------------------
    label_counts = merged["label"].value_counts()
    min_count = label_counts.min()

    print("[BALANCE] Label distribution before balancing:")
    print(label_counts)

    balanced_dfs = []
    for label, count in label_counts.items():
        df_label = merged[merged["label"] == label]
        if count > min_count:
            # Downsample
            df_label = df_label.sample(n=min_count, random_state=42)
        balanced_dfs.append(df_label)

    balanced = pd.concat(balanced_dfs, ignore_index=True)

    print("[BALANCE] Label distribution after balancing:")
    print(balanced["label"].value_counts())

    # Shuffle for randomness
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"[RESULT] Final balanced dataset has {len(balanced)} rows and {len(balanced.columns)} columns")

    # Save dataset
    balanced.to_csv(OUTPUT_FILE, index=False)
    print(f"[SAVED] {OUTPUT_FILE}")
else:
    print("[ERROR] No valid CSVs found")

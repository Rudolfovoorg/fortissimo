import os
import pandas as pd
import streamlit as st
import altair as alt
import db 

# ---------- PATH / DATA ----------
# DATA_FILE = os.path.join("", "load_calculated.csv")
# data_all = pd.read_csv(DATA_FILE,  low_memory=False)


data_all = db.GetEnergyData(4)
data_all["TimeStampMeasured"] = pd.to_datetime(
    data_all["TimeStampMeasured"],
    utc=True,
    errors="coerce"
).dt.tz_convert(None)

# Drop rows where timestamp failed to parse
data_all = data_all.dropna(subset=["TimeStampMeasured"])

# Always sort for correct resampling / ffill
data_all = data_all.sort_values("TimeStampMeasured")

# ---------- FILTER ----------
cutoff = pd.Timestamp("2025-09-01")
subset = data_all.loc[data_all["TimeStampMeasured"] > cutoff, ["TimeStampMeasured", "LoadEnergyCalculated"]].copy()

# ---------- CLEAN SPIKES (HARD CODED) CAREFUL!!! - discuss this with ROBOTINA: ----------
subset.loc[subset["LoadEnergyCalculated"] > 40000, "LoadEnergyCalculated"] = pd.NA
subset["LoadEnergyCalculated"] = subset["LoadEnergyCalculated"].ffill()

# ---------- RESAMPLE 15 MIN ----------
# Set index for resample
energy_ts = subset.set_index("TimeStampMeasured")["LoadEnergyCalculated"]

df_energy_15 = (
    energy_ts
    .resample("15T")
    .sum()
    .rename("LoadEnergyCalculated_15")
    .reset_index()
)

# CAREFUL!!! - discuss this with ROBOTINA: apply the same spike rule after aggregation
df_energy_15.loc[df_energy_15["LoadEnergyCalculated_15"] > 40000, "LoadEnergyCalculated_15"] = pd.NA
df_energy_15["LoadEnergyCalculated_15"] = df_energy_15["LoadEnergyCalculated_15"].ffill()

# Save only final output (if you still want this file)
df_energy_15.to_csv("energy_wh_15.csv", index=False)
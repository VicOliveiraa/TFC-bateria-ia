import pandas as pd
import numpy as np
from pathlib import Path

from config import DERIVED_DIR, MODELS_DIR

def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

def build_ocv_curve(discharge_files, n_bins=200):
    # Junta todos os pontos (SoC, V)
    all_soc = []
    all_v = []

    for f in discharge_files:
        df = pd.read_csv(f)

        # Precisa ter soc_gt e v_bat_v
        if "soc_gt" not in df.columns:
            continue

        all_soc.append(df["soc_gt"].values)
        all_v.append(df["v_bat_v"].values)

    soc = np.concatenate(all_soc)
    v = np.concatenate(all_v)

    # Remove NaN
    mask = np.isfinite(soc) & np.isfinite(v)
    soc = soc[mask]
    v = v[mask]

    # Faz bins uniformes em SoC (0..1)
    bins = np.linspace(0, 1, n_bins + 1)
    soc_mid = (bins[:-1] + bins[1:]) / 2

    v_bin = np.full(n_bins, np.nan)

    for i in range(n_bins):
        m = (soc >= bins[i]) & (soc < bins[i+1])
        if m.any():
            v_bin[i] = np.mean(v[m])

    # Interpola pequenos buracos (se houver)
    s = pd.Series(v_bin)
    v_smooth = s.interpolate(limit_direction="both").rolling(window=7, center=True, min_periods=1).mean().values

    ocv_df = pd.DataFrame({
        "soc": soc_mid,
        "ocv_v": v_smooth
    })

    return ocv_df

def main():
    ensure_dirs()

    # Pega somente descargas (pelo nome)
    discharge_files = sorted(DERIVED_DIR.glob("Descarga*.csv"))

    if not discharge_files:
        print("Nenhum arquivo de descarga encontrado em data/derived/")
        return

    ocv_df = build_ocv_curve(discharge_files, n_bins=200)

    out_path = MODELS_DIR / "ocv_curve.csv"
    ocv_df.to_csv(out_path, index=False)

    print(f"Curva OCV salva em: {out_path}")
    print("Primeiras linhas:")
    print(ocv_df.head())

if __name__ == "__main__":
    main()
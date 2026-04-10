import pandas as pd
from pathlib import Path

from config import PROCESSED_DIR, DERIVED_DIR, MODELS_DIR
from pathlib import Path
import numpy as np

# Reaproveita funções do script 04 copiando o mínimo necessário
def load_ocv_curve(path: Path):
    ocv_df = pd.read_csv(path)
    soc_grid = ocv_df["soc"].to_numpy(dtype=float)
    ocv_grid = ocv_df["ocv_v"].to_numpy(dtype=float)
    return soc_grid, ocv_grid

def ocv_interp(soc_query, soc_grid, ocv_grid):
    soc_query = np.clip(soc_query, 0.0, 1.0)
    return np.interp(soc_query, soc_grid, ocv_grid)

def ocv_derivative(soc_query, soc_grid, ocv_grid):
    eps = 1e-4
    v1 = ocv_interp(soc_query + eps, soc_grid, ocv_grid)
    v0 = ocv_interp(soc_query - eps, soc_grid, ocv_grid)
    return (v1 - v0) / (2 * eps)

from config import C_NOM_AH, R0_OHM, USE_RT_MODEL, T_REF_C, R_REF_OHM, K_OHM_PER_C

def r_of_temp(t_c: float, r0_fallback: float = R0_OHM) -> float:
    if USE_RT_MODEL:
        r = R_REF_OHM + K_OHM_PER_C * (t_c - T_REF_C)
    else:
        r = r0_fallback
    return float(np.clip(r, 0.001, 0.5))

def run_ekf_soc(df: pd.DataFrame, soc_grid, ocv_grid):
    n = len(df)
    soc_est = np.zeros(n, dtype=float)
    v_pred = np.zeros(n, dtype=float)
    v_resid = np.zeros(n, dtype=float)
    r_used = np.zeros(n, dtype=float)

    x = 1.0
    P = 1e-3
    Q = 2e-7
    Rm = 2e-4

    for k in range(n):
        i = float(df["i_a"].iloc[k])
        dt = float(df["dt_s"].iloc[k])
        v_meas = float(df["v_bat_v"].iloc[k])
        t_c = float(df["t_c"].iloc[k])

        x_pred = x - (i * dt) / (C_NOM_AH * 3600.0)
        x_pred = float(np.clip(x_pred, 0.0, 1.0))
        P_pred = P + Q

        ocv = float(ocv_interp(x_pred, soc_grid, ocv_grid))
        rT = r_of_temp(t_c)
        v_model = ocv - i * rT

        H = float(ocv_derivative(x_pred, soc_grid, ocv_grid))
        y = v_meas - v_model
        S = H * P_pred * H + Rm
        K = (P_pred * H) / S if S > 0 else 0.0

        x = x_pred + K * y
        x = float(np.clip(x, 0.0, 1.0))
        P = (1 - K * H) * P_pred

        soc_est[k] = x
        v_pred[k] = v_model
        v_resid[k] = y
        r_used[k] = rT

    out = df.copy()
    out["soc_est"] = soc_est
    out["v_pred"] = v_pred
    out["v_resid"] = v_resid
    out["r_used_ohm"] = r_used
    return out

def main():
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    ocv_path = MODELS_DIR / "ocv_curve.csv"
    if not ocv_path.exists():
        print("models/ocv_curve.csv não encontrado.")
        return
    soc_grid, ocv_grid = load_ocv_curve(ocv_path)

    stress_path = PROCESSED_DIR / "Estresse.csv"
    if not stress_path.exists():
        print("data/processed/StressDischarge.csv não encontrado.")
        return

    df = pd.read_csv(stress_path)
    if "mode" in df.columns:
        df = df[df["mode"].astype(str).str.strip().str.upper() == "DISCH"].copy()

    out = run_ekf_soc(df, soc_grid, ocv_grid)

    out_path = DERIVED_DIR / "Estresse_ekf.csv"
    out.to_csv(out_path, index=False)

    print(f"OK ✅ salvo: {out_path}")

if __name__ == "__main__":
    main()
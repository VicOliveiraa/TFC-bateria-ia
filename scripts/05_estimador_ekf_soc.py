import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    DERIVED_DIR,
    MODELS_DIR,
    RESULTS_TABLES_DIR,
    C_NOM_AH,
    R0_OHM,
    USE_RT_MODEL,
    T_REF_C,
    R_REF_OHM,
    K_OHM_PER_C,
)
MODEL_TAG = "r0"

def load_ocv_curve(path: Path):
    ocv_df = pd.read_csv(path)
    soc_grid = ocv_df["soc"].to_numpy(dtype=float)
    ocv_grid = ocv_df["ocv_v"].to_numpy(dtype=float)
    return soc_grid, ocv_grid


def ocv_interp(soc_query, soc_grid, ocv_grid):
    soc_query = np.clip(soc_query, 0.0, 1.0)
    return np.interp(soc_query, soc_grid, ocv_grid)


def ocv_derivative(soc_query, soc_grid, ocv_grid):
    # derivada numérica central
    eps = 1e-4
    v1 = ocv_interp(soc_query + eps, soc_grid, ocv_grid)
    v0 = ocv_interp(soc_query - eps, soc_grid, ocv_grid)
    return (v1 - v0) / (2 * eps)


def r_of_temp(t_c: float, r0_fallback: float = R0_OHM) -> float:
    """
    Modelo linear:
      R(T) = R_ref + k*(T - T_ref)
    """
    if USE_RT_MODEL:
        r = R_REF_OHM + K_OHM_PER_C * (t_c - T_REF_C)
    else:
        r = r0_fallback

    # segurança: evita valores negativos ou absurdos
    r = float(np.clip(r, 0.001, 0.5))
    return r


def run_ekf_soc(df: pd.DataFrame, soc_grid, ocv_grid, c_nom_ah=C_NOM_AH):
    """
    EKF com estado x = SoC.
    Modelo:
      SoC_k = SoC_{k-1} - (I_k * dt) / (C_nom * 3600)
    Medição:
      V_k = OCV(SoC_k) - I_k * R(T)
    """
    n = len(df)
    soc_est = np.zeros(n, dtype=float)
    v_pred = np.zeros(n, dtype=float)
    v_resid = np.zeros(n, dtype=float)
    r_used = np.zeros(n, dtype=float)

    # Inicialização
    x = 1.0
    P = 1e-3

    # Ruídos (tunable)
    Q = 2e-7    # ruído do processo
    Rm = 2e-4   # ruído da medição (V^2) ~ 14 mV RMS

    for k in range(n):
        i = float(df["i_a"].iloc[k])
        dt = float(df["dt_s"].iloc[k])
        v_meas = float(df["v_bat_v"].iloc[k])
        t_c = float(df["t_c"].iloc[k]) if "t_c" in df.columns else float("nan")

        # ---------- Predição ----------
        x_pred = x - (i * dt) / (c_nom_ah * 3600.0)
        x_pred = float(np.clip(x_pred, 0.0, 1.0))
        P_pred = P + Q

        # ---------- Medição ----------
        ocv = float(ocv_interp(x_pred, soc_grid, ocv_grid))
        rT = r_of_temp(t_c, r0_fallback=R0_OHM) if np.isfinite(t_c) else float(R0_OHM)
        v_model = ocv - i * rT

        # Jacobiano
        H = float(ocv_derivative(x_pred, soc_grid, ocv_grid))

        # Inovação
        y = v_meas - v_model
        S = H * P_pred * H + Rm

        # Ganho de Kalman
        K = (P_pred * H) / S if S > 0 else 0.0

        # Atualização
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
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    ocv_path = MODELS_DIR / "ocv_curve.csv"
    if not ocv_path.exists():
        print("Arquivo models/ocv_curve.csv não encontrado. Rode o 03_ocv_model.py primeiro.")
        return

    soc_grid, ocv_grid = load_ocv_curve(ocv_path)

    discharge_files = sorted(DERIVED_DIR.glob("Descarga*.csv"))
    if not discharge_files:
        print("Nenhum arquivo Descarga.csv encontrado em data/derived/")
        return

    metrics = []

    for f in discharge_files:
        df = pd.read_csv(f)

        df_out = run_ekf_soc(df, soc_grid, ocv_grid)

        # métricas vs soc_gt
        if "soc_gt" in df_out.columns:
            err = df_out["soc_est"] - df_out["soc_gt"]
            mae = float(np.mean(np.abs(err)))
            rmse = float(np.sqrt(np.mean(err**2)))
        else:
            mae, rmse = np.nan, np.nan

        out_path = DERIVED_DIR / f.name.replace(".csv", f"_ekf_{MODEL_TAG}.csv")
        df_out.to_csv(out_path, index=False)

        metrics.append(
            {
                "file": f.name,
                "C_NOM_AH": float(C_NOM_AH),
                "USE_RT_MODEL": bool(USE_RT_MODEL),
                "T_REF_C": float(T_REF_C),
                "R_REF_OHM": float(R_REF_OHM),
                "K_OHM_PER_C": float(K_OHM_PER_C),
                "soc_mae": mae,
                "soc_rmse": rmse,
            }
        )

        print(f"{f.name} -> salvo: {out_path.name} | MAE={mae:.6f} RMSE={rmse:.6f}")

    metrics_df = pd.DataFrame(metrics)
    metrics_path = RESULTS_TABLES_DIR / "soc_ekf_metrics_r0.csv"
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nMétricas salvas em: {metrics_path}")


if __name__ == "__main__":
    main()
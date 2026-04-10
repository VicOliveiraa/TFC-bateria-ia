import numpy as np
import pandas as pd

from config import DERIVED_DIR, RESULTS_TABLES_DIR


def norm_mode(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def rolling_zscore(x: pd.Series, w: int):
    mu = x.rolling(w, min_periods=max(10, w // 4)).mean()
    sd = x.rolling(w, min_periods=max(10, w // 4)).std().replace(0, np.nan)
    return (x - mu) / sd


def add_anomalies_discharge_like(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para arquivos de descarga (inclui stress com EKF):
    Usa v_resid (resíduo do EKF) + dT/dt + overtemp.
    """
    out = df.copy()

    # Parâmetros (tune simples)
    w = 120         # ~120 s em 1 Hz
    z_th = 4.0      # limiar para resíduo
    persist_n = 3   # precisa persistir por 3 amostras

    dT_th = 0.08    # °C/s (≈ 4.8 °C/min) -> bem conservador
    overtemp = 50.0 # °C

    # Elétrica: z-score do resíduo
    if "v_resid" in out.columns:
        z = rolling_zscore(out["v_resid"], w=w)
        elec = (z.abs() > z_th).astype(int)
        elec_persist = (elec.rolling(persist_n, min_periods=persist_n).sum() >= persist_n)
        out["v_resid_z"] = z
        out["anom_elec"] = elec_persist.astype(int)
    else:
        out["v_resid_z"] = np.nan
        out["anom_elec"] = 0

    # Térmica: dT/dt e temperatura máxima
    if "t_c" in out.columns and "dt_s" in out.columns:
        dt = out["dt_s"].replace(0, np.nan)
        dTdt = out["t_c"].diff() / dt
        out["dTdt_c_per_s"] = dTdt

        out["anom_dTdt"] = (dTdt.abs() > dT_th).astype(int)
        out["anom_overtemp"] = (out["t_c"] >= overtemp).astype(int)
    else:
        out["dTdt_c_per_s"] = np.nan
        out["anom_dTdt"] = 0
        out["anom_overtemp"] = 0

    # Flag final
    out["anomaly_flag"] = (
        (out["anom_elec"] == 1) |
        (out["anom_dTdt"] == 1) |
        (out["anom_overtemp"] == 1)
    ).astype(int)

    return out


def add_anomalies_charge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para arquivos de carga *_chg.csv:
    - CC: corrente deveria ser estável -> anomalia se |zscore(dI/dt)| alto
    - CV: corrente deveria cair -> anomalia se a corrente sobe muito ou oscila muito
    - Térmica: dT/dt e overtemp
    """
    out = df.copy()

    # Parâmetros
    w = 120
    z_th_i = 4.0
    persist_n = 3

    dT_th = 0.08
    overtemp = 50.0

    # Derivada de corrente (aprox)
    if "i_a" in out.columns and "dt_s" in out.columns:
        dt = out["dt_s"].replace(0, np.nan)
        didt = out["i_a"].diff() / dt
        out["dIdt_a_per_s"] = didt

        z_i = rolling_zscore(didt.fillna(0.0), w=w)
        spike = (z_i.abs() > z_th_i).astype(int)
        spike_persist = (spike.rolling(persist_n, min_periods=persist_n).sum() >= persist_n)

        # Em CC, spikes são mais suspeitos.
        # Em CV, spikes também, mas vamos adicionar uma regra extra:
        # "corrente subiu muito" (contra o taper esperado)
        if "phase_chg" in out.columns:
            is_cv = out["phase_chg"].astype(str).str.upper() == "CV"
        else:
            is_cv = out["v_bat_v"] >= 4.10 if "v_bat_v" in out.columns else pd.Series(False, index=out.index)

        # aumento anormal de corrente na CV (contra o esperado)
        # (ex.: subida > 0.1 A em um passo)
        cv_rise = (out["i_a"].diff() > 0.10) & is_cv

        out["anom_i_spike"] = spike_persist.astype(int)
        out["anom_cv_rise"] = cv_rise.astype(int)
    else:
        out["dIdt_a_per_s"] = np.nan
        out["anom_i_spike"] = 0
        out["anom_cv_rise"] = 0

    # Térmica
    if "t_c" in out.columns and "dt_s" in out.columns:
        dt = out["dt_s"].replace(0, np.nan)
        dTdt = out["t_c"].diff() / dt
        out["dTdt_c_per_s"] = dTdt
        out["anom_dTdt"] = (dTdt.abs() > dT_th).astype(int)
        out["anom_overtemp"] = (out["t_c"] >= overtemp).astype(int)
    else:
        out["dTdt_c_per_s"] = np.nan
        out["anom_dTdt"] = 0
        out["anom_overtemp"] = 0

    out["anomaly_flag_chg"] = (
        (out["anom_i_spike"] == 1) |
        (out["anom_cv_rise"] == 1) |
        (out["anom_dTdt"] == 1) |
        (out["anom_overtemp"] == 1)
    ).astype(int)

    return out


def main():
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # Arquivos de descarga com EKF
    dis_files = sorted(DERIVED_DIR.glob("NewDischarge*_Certo_ekf_rt.csv"))
    stress_files = sorted(DERIVED_DIR.glob("StressDischarge*_ekf.csv"))
    all_dis = dis_files + stress_files

    # Arquivos de carga derivados
    chg_files = sorted(DERIVED_DIR.glob("NewCharge*_Certo_chg.csv"))

    summary = []

    # --- Descarga / stress ---
    for f in all_dis:
        df = pd.read_csv(f)
        out = add_anomalies_discharge_like(df)
        out.to_csv(f, index=False)

        summary.append({
            "file": f.name,
            "type": "discharge_like",
            "n_samples": int(len(out)),
            "anomaly_total": int(out["anomaly_flag"].sum()),
            "anom_elec": int(out["anom_elec"].sum()),
            "anom_dTdt": int(out["anom_dTdt"].sum()),
            "anom_overtemp": int(out["anom_overtemp"].sum()),
            "max_temp_c": float(out["t_c"].max()) if "t_c" in out.columns else np.nan,
            "max_abs_v_resid_z": float(np.nanmax(np.abs(out["v_resid_z"]))) if "v_resid_z" in out.columns else np.nan,
        })

        print(f"{f.name} -> OK (anomaly_flag) total={int(out['anomaly_flag'].sum())}")

    # --- Carga ---
    for f in chg_files:
        df = pd.read_csv(f)
        out = add_anomalies_charge(df)
        out.to_csv(f, index=False)

        summary.append({
            "file": f.name,
            "type": "charge",
            "n_samples": int(len(out)),
            "anomaly_total": int(out["anomaly_flag_chg"].sum()),
            "anom_i_spike": int(out["anom_i_spike"].sum()),
            "anom_cv_rise": int(out["anom_cv_rise"].sum()),
            "anom_dTdt": int(out["anom_dTdt"].sum()),
            "anom_overtemp": int(out["anom_overtemp"].sum()),
            "max_temp_c": float(out["t_c"].max()) if "t_c" in out.columns else np.nan,
        })

        print(f"{f.name} -> OK (anomaly_flag_chg) total={int(out['anomaly_flag_chg'].sum())}")

    # salva resumo
    sum_df = pd.DataFrame(summary)
    out_path = RESULTS_TABLES_DIR / "anomaly_summary.csv"
    sum_df.to_csv(out_path, index=False)
    print(f"\nOK ✅ Resumo salvo em: {out_path}")


if __name__ == "__main__":
    main()
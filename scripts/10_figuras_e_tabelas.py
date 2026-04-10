import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Optional

from config import DERIVED_DIR, MODELS_DIR, RESULTS_TABLES_DIR, C_NOM_AH


# =========================
# Helpers
# =========================

def ensure_dirs() -> Path:
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    figures_dir = RESULTS_TABLES_DIR.parent / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    return figures_dir


def safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def pick_time_column(df: pd.DataFrame) -> Optional[str]:
    for col in ["time_s", "t_s", "t_sec", "t"]:
        if col in df.columns:
            return col
    return None


def remaining_time_true_minutes(df: pd.DataFrame) -> Optional[pd.Series]:
    tcol = pick_time_column(df)
    if tcol is None:
        return None
    t = df[tcol].astype(float)
    return (t.iloc[-1] - t) / 60.0


def compute_soc_metrics(df: pd.DataFrame, soc_col: str, gt_col: str = "soc_gt") -> dict:
    err = df[soc_col] - df[gt_col]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    return {"mae": mae, "rmse": rmse}


def x_axis(df: pd.DataFrame):
    tcol = pick_time_column(df)
    if tcol is None:
        return np.arange(len(df)), "amostra"
    x = df[tcol].astype(float) / 3600.0
    return x, "tempo (h)"


def save_fig(fig, ax, path: Path):
    # Força legenda se existir qualquer série com label
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# =========================
# Figures
# =========================

def plot_soc_compare(df: pd.DataFrame, file_tag: str, out_dir: Path):
    if "soc_gt" not in df.columns or "soc_est" not in df.columns:
        return

    x, xlab = x_axis(df)

    fig, ax = plt.subplots()
    ax.plot(x, df["soc_gt"], label="SoC referência (GT)")
    ax.plot(x, df["soc_est"], label="SoC estimado (EKF)")
    ax.set_xlabel(xlab)
    ax.set_ylabel("SoC")
    ax.set_title(f"SoC (GT vs Est) - {file_tag}")
    save_fig(fig, ax, out_dir / f"soc_gt_vs_est__{file_tag}.png")

    fig, ax = plt.subplots()
    ax.plot(x, (df["soc_est"] - df["soc_gt"]))
    ax.set_xlabel(xlab)
    ax.set_ylabel("erro SoC (est - gt)")
    ax.set_title(f"Erro de SoC - {file_tag}")
    save_fig(fig, ax, out_dir / f"soc_error__{file_tag}.png")


def plot_voltage_residuals(df: pd.DataFrame, file_tag: str, out_dir: Path):
    if "v_bat_v" not in df.columns or "v_pred" not in df.columns or "v_resid" not in df.columns:
        return

    x, xlab = x_axis(df)

    fig, ax = plt.subplots()
    ax.plot(x, df["v_bat_v"], label="Tensão medida")
    ax.plot(x, df["v_pred"], label="Tensão prevista (modelo)")
    ax.set_xlabel(xlab)
    ax.set_ylabel("Tensão (V)")
    ax.set_title(f"Tensão medida vs prevista - {file_tag}")
    save_fig(fig, ax, out_dir / f"voltage_meas_vs_pred__{file_tag}.png")

    fig, ax = plt.subplots()
    ax.plot(x, df["v_resid"])
    ax.set_xlabel(xlab)
    ax.set_ylabel("resíduo (V)")
    ax.set_title(f"Resíduo de tensão (v_resid) - {file_tag}")
    save_fig(fig, ax, out_dir / f"v_resid__{file_tag}.png")


def plot_remaining_time(df: pd.DataFrame, file_tag: str, out_dir: Path, col_pred: str):
    t_true = remaining_time_true_minutes(df)
    if t_true is None or col_pred not in df.columns:
        return

    x, xlab = x_axis(df)

    fig, ax = plt.subplots()
    ax.plot(x, t_true, label="Tempo restante real")
    ax.plot(x, df[col_pred], label="Tempo restante estimado")
    ax.set_xlabel(xlab)
    ax.set_ylabel("tempo restante (min)")
    ax.set_title(f"Tempo restante: real vs estimado - {file_tag}")
    save_fig(fig, ax, out_dir / f"t_remaining__{file_tag}.png")

    fig, ax = plt.subplots()
    ax.plot(x, (df[col_pred] - t_true))
    ax.set_xlabel(xlab)
    ax.set_ylabel("erro tempo restante (min)")
    ax.set_title(f"Erro tempo restante - {file_tag}")
    save_fig(fig, ax, out_dir / f"t_remaining_error__{file_tag}.png")


def plot_rt_scatter(out_dir: Path):
    events_path = RESULTS_TABLES_DIR / "stress_rT_events.csv"
    params_path = MODELS_DIR / "rT_params.csv"

    ev = safe_read_csv(events_path)
    params = safe_read_csv(params_path)

    if ev is None or len(ev) == 0 or params is None or len(params) == 0:
        return

    p = params.iloc[0]
    T_ref = float(p["T_ref_C"])
    R_ref = float(p["R_ref_ohm"])
    k = float(p["k_ohm_per_C"])

    T = ev["T_event_C"].astype(float).to_numpy()
    R = ev["R_ohm"].astype(float).to_numpy()

    T_line = np.linspace(np.min(T), np.max(T), 200)
    R_line = R_ref + k * (T_line - T_ref)

    fig, ax = plt.subplots()
    ax.scatter(T, R, s=10, label="Eventos extraídos (stress)")
    ax.plot(T_line, R_line, label="Ajuste linear R(T)")
    ax.set_xlabel("Temperatura (°C)")
    ax.set_ylabel("R (ohm)")
    ax.set_title("R(T) a partir do ensaio de stress")
    save_fig(fig, ax, out_dir / "rt_scatter_fit.png")


# =========================
# Consolidation (tables)
# =========================

def consolidate_soc_metrics(out_tables: Path) -> Path:
    rows = []

    rt_files = sorted(DERIVED_DIR.glob("NewDischarge*_Certo_ekf_rt.csv"))
    r0_files = sorted(DERIVED_DIR.glob("NewDischarge*_Certo_ekf_r0.csv"))
    if not rt_files:
        rt_files = sorted(DERIVED_DIR.glob("NewDischarge*_Certo_ekf.csv"))

    def base_name(p: Path) -> str:
        name = p.name
        for suf in ["_ekf_rt.csv", "_ekf_r0.csv", "_ekf.csv"]:
            if name.endswith(suf):
                return name.replace(suf, "")
        return p.stem

    rt_map = {base_name(p): p for p in rt_files}
    r0_map = {base_name(p): p for p in r0_files}
    bases = sorted(set(list(rt_map.keys()) + list(r0_map.keys())))

    for b in bases:
        row = {"base": b}

        # RT
        if b in rt_map:
            df = safe_read_csv(rt_map[b])
            if df is not None and "soc_gt" in df.columns and "soc_est" in df.columns:
                m = compute_soc_metrics(df, "soc_est", "soc_gt")
                row["mae_rt"] = m["mae"]
                row["rmse_rt"] = m["rmse"]
            else:
                row["mae_rt"] = np.nan
                row["rmse_rt"] = np.nan
        else:
            row["mae_rt"] = np.nan
            row["rmse_rt"] = np.nan

        # R0
        if b in r0_map:
            df = safe_read_csv(r0_map[b])
            if df is not None and "soc_gt" in df.columns and "soc_est" in df.columns:
                m = compute_soc_metrics(df, "soc_est", "soc_gt")
                row["mae_r0"] = m["mae"]
                row["rmse_r0"] = m["rmse"]
            else:
                row["mae_r0"] = np.nan
                row["rmse_r0"] = np.nan
        else:
            row["mae_r0"] = np.nan
            row["rmse_r0"] = np.nan

        # melhorias (%)
        if np.isfinite(row["mae_r0"]) and np.isfinite(row["mae_rt"]) and row["mae_r0"] > 0:
            row["mae_improvement_pct"] = 100.0 * (row["mae_r0"] - row["mae_rt"]) / row["mae_r0"]
        else:
            row["mae_improvement_pct"] = np.nan

        if np.isfinite(row["rmse_r0"]) and np.isfinite(row["rmse_rt"]) and row["rmse_r0"] > 0:
            row["rmse_improvement_pct"] = 100.0 * (row["rmse_r0"] - row["rmse_rt"]) / row["rmse_r0"]
        else:
            row["rmse_improvement_pct"] = np.nan

        rows.append(row)

    out = pd.DataFrame(rows)
    out_path = out_tables / "soc_metrics_compare.csv"
    out.to_csv(out_path, index=False)
    return out_path


def consolidate_remaining_time_metrics(out_tables: Path) -> Path:
    rows = []

    # Descarga: preferir RT final
    dis_files = sorted(DERIVED_DIR.glob("Descarga*_ekf_rt.csv"))
    if not dis_files:
        dis_files = sorted(DERIVED_DIR.glob("Descarga*_ekf.csv"))

    for f in dis_files:
        df = safe_read_csv(f)
        if df is None or "t_rem_min_dis" not in df.columns:
            continue
        t_true = remaining_time_true_minutes(df)
        if t_true is None:
            continue
        pred = df["t_rem_min_dis"].astype(float)
        err = pred - t_true
        rows.append({
            "file": f.name,
            "type": "discharge",
            "mae_min": float(np.mean(np.abs(err))),
            "rmse_min": float(np.sqrt(np.mean(err ** 2))),
        })

    # Carga
    chg_files = sorted(DERIVED_DIR.glob("Carga*_chg.csv"))
    for f in chg_files:
        df = safe_read_csv(f)
        if df is None or "t_rem_min_chg" not in df.columns:
            continue
        t_true = remaining_time_true_minutes(df)
        if t_true is None:
            continue
        pred = df["t_rem_min_chg"].astype(float)
        err = pred - t_true
        rows.append({
            "file": f.name,
            "type": "charge",
            "mae_min": float(np.mean(np.abs(err))),
            "rmse_min": float(np.sqrt(np.mean(err ** 2))),
        })

    out = pd.DataFrame(rows)
    out_path = out_tables / "remaining_time_metrics.csv"
    out.to_csv(out_path, index=False)
    return out_path


def consolidate_global_summary(out_tables: Path) -> Path:
    cap_path = DERIVED_DIR / "capacity_summary.csv"
    chg_sum_path = RESULTS_TABLES_DIR / "charge_summary.csv"

    cap = safe_read_csv(cap_path)
    chg = safe_read_csv(chg_sum_path)

    if cap is not None and "ah_total" in cap.columns:
        dis_mean = float(cap["ah_total"].mean())
        dis_std = float(cap["ah_total"].std(ddof=0))
    else:
        dis_mean = np.nan
        dis_std = np.nan

    if chg is not None and "ah_total_chg" in chg.columns:
        chg_mean = float(chg["ah_total_chg"].mean())
        chg_std = float(chg["ah_total_chg"].std(ddof=0))
    else:
        chg_mean = np.nan
        chg_std = np.nan

    eff = (dis_mean / chg_mean) if (np.isfinite(dis_mean) and np.isfinite(chg_mean) and chg_mean > 0) else np.nan

    rtp = safe_read_csv(MODELS_DIR / "rT_params.csv")
    p = rtp.iloc[0].to_dict() if (rtp is not None and len(rtp) > 0) else {}

    out = pd.DataFrame([{
        "C_NOM_AH_used": float(C_NOM_AH),
        "discharge_capacity_mean_ah": dis_mean,
        "discharge_capacity_std_ah": dis_std,
        "charge_inserted_mean_ah": chg_mean,
        "charge_inserted_std_ah": chg_std,
        "coulombic_efficiency_approx": float(eff) if np.isfinite(eff) else np.nan,
        "R_ref_ohm": p.get("R_ref_ohm", np.nan),
        "T_ref_C": p.get("T_ref_C", np.nan),
        "k_ohm_per_C": p.get("k_ohm_per_C", np.nan),
        "R2": p.get("R2", np.nan),
        "n_events_rt": p.get("n_events", np.nan),
        "T_min_C": p.get("T_min_C", np.nan),
        "T_max_C": p.get("T_max_C", np.nan),
    }])

    out_path = out_tables / "global_summary.csv"
    out.to_csv(out_path, index=False)
    return out_path


# =========================
# Figures batch
# =========================

def generate_figures(figures_dir: Path):
    # Descargas
    rt_files = sorted(DERIVED_DIR.glob("Descarga*_ekf_rt.csv"))
    r0_files = sorted(DERIVED_DIR.glob("Descarga*_ekf_r0.csv"))
    if not rt_files:
        rt_files = sorted(DERIVED_DIR.glob("Descarga*_ekf.csv"))

    for f in rt_files:
        df = safe_read_csv(f)
        if df is None:
            continue
        tag = f.stem
        plot_soc_compare(df, tag, figures_dir)
        plot_voltage_residuals(df, tag, figures_dir)
        plot_remaining_time(df, tag, figures_dir, "t_rem_min_dis")

    for f in r0_files:
        df = safe_read_csv(f)
        if df is None:
            continue
        tag = f.stem
        plot_soc_compare(df, tag, figures_dir)
        plot_voltage_residuals(df, tag, figures_dir)

    # Stress
    stress_candidates = (
        sorted(DERIVED_DIR.glob("Estresse*_ekf_rt.csv")) +
        sorted(DERIVED_DIR.glob("Estresse*_ekf_r0.csv")) +
        sorted(DERIVED_DIR.glob("Estresse_ekf.csv"))
    )

    for f in stress_candidates:
        df = safe_read_csv(f)
        if df is None:
            continue
        tag = f.stem
        plot_voltage_residuals(df, tag, figures_dir)
        plot_remaining_time(df, tag, figures_dir, "t_rem_min_dis")

        if "r_used_ohm" in df.columns:
            x, xlab = x_axis(df)
            fig, ax = plt.subplots()
            ax.plot(x, df["r_used_ohm"])
            ax.set_xlabel(xlab)
            ax.set_ylabel("R usada (ohm)")
            ax.set_title(f"R usada no EKF - {tag}")
            save_fig(fig, ax, figures_dir / f"r_used__{tag}.png")

        if "t_c" in df.columns:
            x, xlab = x_axis(df)
            fig, ax = plt.subplots()
            ax.plot(x, df["t_c"])
            ax.set_xlabel(xlab)
            ax.set_ylabel("Temperatura (°C)")
            ax.set_title(f"Temperatura - {tag}")
            save_fig(fig, ax, figures_dir / f"temp__{tag}.png")

    # Cargas
    for f in sorted(DERIVED_DIR.glob("Carga*_chg.csv")):
        df = safe_read_csv(f)
        if df is None:
            continue
        tag = f.stem
        x, xlab = x_axis(df)

        if "soc_gt_chg" in df.columns:
            fig, ax = plt.subplots()
            ax.plot(x, df["soc_gt_chg"], label="SoC referência (carga)")
            if "soc_est_chg" in df.columns:
                ax.plot(x, df["soc_est_chg"], label="SoC estimado (coulomb counting)")
            ax.set_xlabel(xlab)
            ax.set_ylabel("SoC")
            ax.set_title(f"SoC na carga - {tag}")
            save_fig(fig, ax, figures_dir / f"charge_soc__{tag}.png")

        if "i_a" in df.columns:
            fig, ax = plt.subplots()
            ax.plot(x, df["i_a"])
            ax.set_xlabel(xlab)
            ax.set_ylabel("Corrente (A)")
            ax.set_title(f"Corrente na carga - {tag}")
            save_fig(fig, ax, figures_dir / f"charge_current__{tag}.png")

        if "v_bat_v" in df.columns:
            fig, ax = plt.subplots()
            ax.plot(x, df["v_bat_v"])
            ax.set_xlabel(xlab)
            ax.set_ylabel("Tensão (V)")
            ax.set_title(f"Tensão na carga - {tag}")
            save_fig(fig, ax, figures_dir / f"charge_voltage__{tag}.png")

        if "t_rem_min_chg" in df.columns:
            plot_remaining_time(df, tag, figures_dir, "t_rem_min_chg")

    # R(T) scatter + fit
    plot_rt_scatter(figures_dir)


# =========================
# Main
# =========================

def main():
    figures_dir = ensure_dirs()

    p1 = consolidate_soc_metrics(RESULTS_TABLES_DIR)
    p2 = consolidate_remaining_time_metrics(RESULTS_TABLES_DIR)
    p3 = consolidate_global_summary(RESULTS_TABLES_DIR)

    anom = safe_read_csv(RESULTS_TABLES_DIR / "anomaly_summary.csv")
    if anom is not None:
        anom.to_csv(RESULTS_TABLES_DIR / "anomaly_summary_final.csv", index=False)

    generate_figures(figures_dir)

    print("OK ✅ Resultados consolidados e figuras geradas.")
    print(f"- Tabelas: {RESULTS_TABLES_DIR}")
    print(f"- Figuras: {figures_dir}")
    print(f"- Gerado: {p1.name}, {p2.name}, {p3.name}")


if __name__ == "__main__":
    main()
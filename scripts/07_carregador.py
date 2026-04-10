import numpy as np
import pandas as pd
from pathlib import Path

from config import PROCESSED_DIR, DERIVED_DIR, RESULTS_TABLES_DIR, C_NOM_AH


def ensure_dirs():
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def norm_mode(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def detect_cv_start(
    df: pd.DataFrame,
    v_cv_thresh: float = 4.10,
    i_drop_frac: float = 0.92,
    min_cv_duration_s: float = 120.0,
):
    """
    Detecta o início da fase CV.
    Heurística robusta:
      - CV exige V >= v_cv_thresh
      - e corrente começa a cair (abaixo de frac do "I_CC" típico do começo)
      - e condição se mantém por pelo menos min_cv_duration_s

    Retorna:
      idx_cv_start (int ou None), t_cv_start_s (float ou None)
    """
    if len(df) < 10:
        return None, None

    # corrente típica do trecho inicial (CC)
    n0 = min(300, len(df))  # ~ primeiros 5 min se 1 Hz
    i0 = df["i_a"].iloc[:n0].median()

    # condição de CV candidata
    cond = (df["v_bat_v"] >= v_cv_thresh) & (df["i_a"] <= i0 * i_drop_frac)

    if not cond.any():
        # fallback: se só V>=threshold já caracteriza CV em alguns carregadores
        cond2 = df["v_bat_v"] >= v_cv_thresh
        if not cond2.any():
            return None, None
        # exige persistência
        w = max(10, int(min_cv_duration_s / max(df["dt_s"].median(), 1e-3)))
        pers = cond2.rolling(w, min_periods=w).sum() >= w
        if pers.any():
            idx = int(np.where(pers.to_numpy())[0][0])
            return idx, float(df["time_s"].iloc[idx])
        return None, None

    # exige persistência por um tempo mínimo
    dt_med = float(df["dt_s"].median()) if "dt_s" in df.columns else 1.0
    w = max(10, int(min_cv_duration_s / max(dt_med, 1e-3)))  # nº de amostras

    pers = cond.rolling(w, min_periods=w).sum() >= w
    if not pers.any():
        return None, None

    idx = int(np.where(pers.to_numpy())[0][0])
    return idx, float(df["time_s"].iloc[idx])


def process_one_charge(file_path: Path):
    df = pd.read_csv(file_path)

    # filtra CHG
    if "mode" in df.columns:
        m = norm_mode(df["mode"])
        df = df[m == "CHG"].copy()

    if len(df) == 0:
        return None, None

    # Checagens mínimas
    required = ["time_s", "dt_s", "i_a", "v_bat_v", "t_c"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{file_path.name}: faltam colunas {missing}")

    df = df.sort_values("time_s").reset_index(drop=True)

    # integrações
    df["ah_accum_chg"] = (df["i_a"] * df["dt_s"]).cumsum() / 3600.0
    df["wh_accum_chg"] = (df["i_a"] * df["v_bat_v"] * df["dt_s"]).cumsum() / 3600.0

    ah_total = float(df["ah_accum_chg"].iloc[-1])
    wh_total = float(df["wh_accum_chg"].iloc[-1])

    # SoC de referência do próprio ensaio (0->1)
    # (boa para comparar perfis entre ensaios e para validar tempo restante)
    if ah_total > 1e-9:
        df["soc_gt_chg"] = (df["ah_accum_chg"] / ah_total).clip(0.0, 1.0)
    else:
        df["soc_gt_chg"] = 0.0

    # SoC estimado "online" usando capacidade nominal do projeto
    df["soc_est_chg"] = (df["ah_accum_chg"] / float(C_NOM_AH)).clip(0.0, 1.0)

    # detecta CC/CV
    idx_cv, t_cv = detect_cv_start(df, v_cv_thresh=4.10, i_drop_frac=0.92, min_cv_duration_s=120.0)

    df["phase_chg"] = "CC"
    if idx_cv is not None:
        df.loc[idx_cv:, "phase_chg"] = "CV"

    # resumo
    summary = {
        "file": file_path.name,
        "n_samples": int(len(df)),
        "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]),
        "duration_h": float((df["time_s"].iloc[-1] - df["time_s"].iloc[0]) / 3600.0),
        "i_mean_a": float(df["i_a"].mean()),
        "i_start_a": float(df["i_a"].iloc[0]),
        "i_end_a": float(df["i_a"].iloc[-1]),
        "v_start_v": float(df["v_bat_v"].iloc[0]),
        "v_end_v": float(df["v_bat_v"].iloc[-1]),
        "t_start_c": float(df["t_c"].iloc[0]),
        "t_end_c": float(df["t_c"].iloc[-1]),
        "ah_total_chg": ah_total,
        "wh_total_chg": wh_total,
        "cv_start_s": float(t_cv) if t_cv is not None else np.nan,
        "cv_detected": bool(t_cv is not None),
    }

    return df, summary


def main():
    ensure_dirs()

    files = sorted(PROCESSED_DIR.glob("Carga*.csv"))
    if not files:
        print("Nenhum NewCharge*_Certo.csv encontrado em data/processed/")
        return

    summaries = []

    for f in files:
        df_out, summary = process_one_charge(f)
        if df_out is None:
            print(f"{f.name}: sem dados CHG, pulando.")
            continue

        out_path = DERIVED_DIR / f.name.replace(".csv", "_chg.csv")
        df_out.to_csv(out_path, index=False)
        summaries.append(summary)

        cv_msg = f"CV@{summary['cv_start_s']:.0f}s" if summary["cv_detected"] else "CV não detectada"
        print(f"{f.name} -> salvo: {out_path.name} | Ah={summary['ah_total_chg']:.4f} | {cv_msg}")

    if summaries:
        sum_df = pd.DataFrame(summaries)
        sum_path = RESULTS_TABLES_DIR / "charge_summary.csv"
        sum_df.to_csv(sum_path, index=False)
        print(f"\nResumo salvo em: {sum_path}")

    print("\nOK ✅ Script 08 concluído.")


if __name__ == "__main__":
    main()
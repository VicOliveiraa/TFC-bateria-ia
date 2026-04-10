
import numpy as np
import pandas as pd
from pathlib import Path

from config import DERIVED_DIR, MODELS_DIR, RESULTS_TABLES_DIR, C_NOM_AH

DISCH_I_MEAN_W = 60
CHG_I_MEAN_W = 60
CHG_TREM_SMOOTH_W = 21

# carga
SOC_LOOKUP_MAX = 0.985      # faixa em que soc_est_chg ainda e util
SOC_TAIL_BLEND_START = 0.99 # comeca a dar peso ao modelo de corrente
V_TAIL_BLEND_START = 4.195  # idem pela tensao
V_TAIL_BLEND_END = 4.205


def ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def norm_mode(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper()


def safe_rolling_mean(x: pd.Series, w: int) -> pd.Series:
    return x.rolling(w, min_periods=1).mean()


def running_min(x: pd.Series | np.ndarray) -> pd.Series:
    s = pd.Series(x, dtype=float)
    return s.cummin()


def add_remaining_time_discharge(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "soc_est" not in out.columns:
        raise ValueError("Arquivo nao tem 'soc_est'. Rode o EKF antes.")
    if "i_a" not in out.columns:
        raise ValueError("Arquivo nao tem 'i_a'.")

    i_mean = safe_rolling_mean(out["i_a"].astype(float).abs(), DISCH_I_MEAN_W)
    ah_remaining = out["soc_est"].clip(0, 1) * float(C_NOM_AH)
    t_rem_h = ah_remaining / i_mean.replace(0, np.nan)
    out["i_mean_60s"] = i_mean
    out["t_rem_min_dis"] = (t_rem_h * 60.0).clip(lower=0)
    return out


def compute_soc_est_chg_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "soc_est_chg" in out.columns:
        out["soc_est_chg"] = out["soc_est_chg"].astype(float).clip(0.0, 1.0)
        return out

    if "dt_s" not in out.columns or "i_a" not in out.columns:
        raise ValueError("Para criar soc_est_chg preciso de 'dt_s' e 'i_a'.")

    i_pos = out["i_a"].astype(float).abs()
    ah_accum = (i_pos * out["dt_s"].astype(float)).cumsum() / 3600.0
    total_ah_test = float(ah_accum.iloc[-1])

    out["ah_accum_chg"] = ah_accum
    if total_ah_test <= 1e-12:
        out["soc_est_chg"] = 0.0
    else:
        out["soc_est_chg"] = (ah_accum / total_ah_test).clip(0.0, 1.0)
    return out


def build_monotonic_lookup(x: pd.Series, y: pd.Series, nbins: int = 240, increasing: bool = True) -> pd.DataFrame | None:
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"),
                      "y": pd.to_numeric(y, errors="coerce")}).dropna()

    if len(d) < 20:
        return None

    d = d.sort_values("x").reset_index(drop=True)

    x_min = float(d["x"].min())
    x_max = float(d["x"].max())
    if not np.isfinite(x_min) or not np.isfinite(x_max) or x_max <= x_min:
        return None

    bins = np.linspace(x_min, x_max, nbins + 1)
    x_mid = (bins[:-1] + bins[1:]) / 2.0
    y_bin = np.full(nbins, np.nan)

    for k in range(nbins):
        if k < nbins - 1:
            m = (d["x"] >= bins[k]) & (d["x"] < bins[k + 1])
        else:
            m = (d["x"] >= bins[k]) & (d["x"] <= bins[k + 1])

        if m.any():
            y_bin[k] = float(np.median(d.loc[m, "y"]))

    s = pd.Series(y_bin).interpolate(limit_direction="both")
    y_smooth = s.rolling(window=7, center=True, min_periods=1).mean().to_numpy()

    if increasing:
        y_mono = np.maximum.accumulate(y_smooth)
    else:
        y_mono = np.minimum.accumulate(y_smooth)

    return pd.DataFrame({"x": x_mid, "y": y_mono})


def interp_lookup(x_now: pd.Series | np.ndarray, lookup_df: pd.DataFrame) -> np.ndarray:
    xg = lookup_df["x"].to_numpy(dtype=float)
    yg = lookup_df["y"].to_numpy(dtype=float)
    x = np.asarray(x_now, dtype=float)
    x = np.clip(x, xg.min(), xg.max())
    return np.interp(x, xg, yg)


def collect_charge_training_rows(charge_files: list[Path], exclude_name: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows_soc = []
    rows_i = []

    for f in charge_files:
        if exclude_name is not None and f.name == exclude_name and len(charge_files) > 1:
            continue

        try:
            d = pd.read_csv(f)
        except Exception:
            continue

        if "mode" in d.columns:
            m = norm_mode(d["mode"])
            d = d[m == "CHG"].copy()

        if len(d) < 50:
            continue
        if "time_s" not in d.columns:
            continue

        d = compute_soc_est_chg_if_missing(d)

        if "t_rem_true_min" not in d.columns:
            t_true = (float(d["time_s"].iloc[-1]) - d["time_s"].astype(float)) / 60.0
        else:
            t_true = d["t_rem_true_min"].astype(float)

        d["i_mean_60s"] = safe_rolling_mean(d["i_a"].astype(float).abs(), CHG_I_MEAN_W)

        # lookup principal: SoC -> tempo restante (somente antes da saturacao)
        m_soc = d["soc_est_chg"].astype(float) < SOC_LOOKUP_MAX
        if m_soc.sum() >= 20:
            rows_soc.append(pd.DataFrame({
                "soc_est_chg": d.loc[m_soc, "soc_est_chg"].astype(float).values,
                "t_rem_true_min": t_true.loc[m_soc].astype(float).values
            }))

        # lookup de cauda CV: corrente media -> tempo restante
        m_tail = (d["soc_est_chg"].astype(float) >= SOC_TAIL_BLEND_START) | (d["v_bat_v"].astype(float) >= V_TAIL_BLEND_START)
        if m_tail.sum() >= 20:
            rows_i.append(pd.DataFrame({
                "i_mean_60s": d.loc[m_tail, "i_mean_60s"].astype(float).values,
                "t_rem_true_min": t_true.loc[m_tail].astype(float).values
            }))

    soc_rows = pd.concat(rows_soc, ignore_index=True) if rows_soc else pd.DataFrame(columns=["soc_est_chg", "t_rem_true_min"])
    i_rows = pd.concat(rows_i, ignore_index=True) if rows_i else pd.DataFrame(columns=["i_mean_60s", "t_rem_true_min"])
    return soc_rows, i_rows


def add_remaining_time_charge(df: pd.DataFrame, soc_lookup: pd.DataFrame | None, tail_lookup: pd.DataFrame | None) -> pd.DataFrame:
    out = df.copy()

    if "mode" in out.columns:
        m = norm_mode(out["mode"])
        out = out[m == "CHG"].copy()

    if len(out) == 0:
        return df.copy()

    out = compute_soc_est_chg_if_missing(out)

    if "time_s" not in out.columns or "i_a" not in out.columns or "v_bat_v" not in out.columns:
        raise ValueError("Carga precisa de 'time_s', 'i_a' e 'v_bat_v'.")

    out["i_mean_60s"] = safe_rolling_mean(out["i_a"].astype(float).abs(), CHG_I_MEAN_W)

    # fallback: se o arquivo tiver t_rem_true_min e nao houver treino suficiente
    if soc_lookup is None and "t_rem_true_min" in out.columns:
        t_est = out["t_rem_true_min"].astype(float).copy()
    else:
        if soc_lookup is None:
            raise ValueError("Nao foi possivel montar lookup SoC->tempo para carga.")

        t_soc = interp_lookup(out["soc_est_chg"].astype(float), soc_lookup)

        if tail_lookup is not None:
            t_tail = interp_lookup(out["i_mean_60s"].astype(float), tail_lookup)
        else:
            t_tail = t_soc.copy()

        w_soc = np.clip((out["soc_est_chg"].astype(float) - SOC_TAIL_BLEND_START) / max(1e-6, (1.0 - SOC_TAIL_BLEND_START)), 0.0, 1.0)
        w_v = np.clip((out["v_bat_v"].astype(float) - V_TAIL_BLEND_START) / max(1e-6, (V_TAIL_BLEND_END - V_TAIL_BLEND_START)), 0.0, 1.0)
        w_tail = np.maximum(w_soc, w_v)

        t_est = (1.0 - w_tail) * t_soc + w_tail * t_tail
        t_est = pd.Series(t_est, index=out.index)

    # suaviza e garante forma fisica
    t_est = safe_rolling_mean(pd.Series(t_est, index=out.index), CHG_TREM_SMOOTH_W)
    t_est = running_min(t_est)

    if len(t_est) > 0:
        # ancora o fim; nao zera cedo
        t_est.iloc[-1] = 0.0

    out["t_rem_min_chg_raw"] = t_est
    out["t_rem_min_chg"] = t_est.clip(lower=0)
    out["phase_chg"] = "CHG"
    return out


def main():
    ensure_dirs()

    # descarga
    dis_files = sorted(DERIVED_DIR.glob("Descarga*_ekf_rt.csv"))
    stress_files = sorted(DERIVED_DIR.glob("Estresse*_ekf.csv"))
    for f in dis_files + stress_files:
        df = pd.read_csv(f)
        try:
            out = add_remaining_time_discharge(df)
            out.to_csv(f, index=False)
            print(f"{f.name} -> OK (t_rem_min_dis)")
        except Exception as e:
            print(f"{f.name} -> ERRO (descarga): {e}")

    # carga
    chg_files = sorted(DERIVED_DIR.glob("Carga*_chg.csv"))
    if not chg_files:
        chg_files = sorted(DERIVED_DIR.glob("NewCharge*_chg.csv"))

    for f in chg_files:
        df = pd.read_csv(f)
        try:
            soc_rows, i_rows = collect_charge_training_rows(chg_files, exclude_name=f.name)
            if soc_rows.empty and i_rows.empty:
                # ultimo fallback: usa o proprio arquivo para montar o treino
                soc_rows, i_rows = collect_charge_training_rows([f], exclude_name=None)

            soc_lookup = None if soc_rows.empty else build_monotonic_lookup(
                soc_rows["soc_est_chg"], soc_rows["t_rem_true_min"], nbins=240, increasing=False
            )
            tail_lookup = None if i_rows.empty else build_monotonic_lookup(
                i_rows["i_mean_60s"], i_rows["t_rem_true_min"], nbins=240, increasing=True
            )

            out = add_remaining_time_charge(df, soc_lookup=soc_lookup, tail_lookup=tail_lookup)
            out.to_csv(f, index=False)
            print(f"{f.name} -> OK (t_rem_min_chg)")
        except Exception as e:
            print(f"{f.name} -> ERRO (carga): {e}")

    print("OK Script 08 concluido.")


if __name__ == "__main__":
    main()

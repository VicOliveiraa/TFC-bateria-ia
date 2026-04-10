import numpy as np
import pandas as pd

from config import RAW_DIR, RESULTS_TABLES_DIR, MODELS_DIR


def ensure_dirs():
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def normalize_mode(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.upper()


def fit_linear_rT(events_df: pd.DataFrame) -> dict:
    # Ajusta R = a + b*T (mínimos quadrados)
    T = events_df["T_event_C"].to_numpy(dtype=float)
    R = events_df["R_ohm"].to_numpy(dtype=float)

    A = np.vstack([np.ones_like(T), T]).T
    (a, b), *_ = np.linalg.lstsq(A, R, rcond=None)

    T_ref = float(np.mean(T))
    R_ref = float(a + b * T_ref)
    k = float(b)

    R_pred = a + b * T
    ss_res = float(np.sum((R - R_pred) ** 2))
    ss_tot = float(np.sum((R - np.mean(R)) ** 2)) if len(R) > 1 else 0.0
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return {
        "n_events": int(len(R)),
        "T_ref_C": T_ref,
        "R_ref_ohm": R_ref,
        "k_ohm_per_C": k,
        "R2": float(r2),
        "R_median_ohm": float(np.median(R)),
        "R_mean_ohm": float(np.mean(R)),
        "R_std_ohm": float(np.std(R)),
        "T_min_C": float(np.min(T)),
        "T_max_C": float(np.max(T)),
    }


def main():
    ensure_dirs()

    stress_path = RAW_DIR / "Estresse.csv"
    if not stress_path.exists():
        print(f"Não encontrei {stress_path}")
        return

    df = pd.read_csv(stress_path)

    # Filtra somente DISCH se existir essa classe
    if "mode" in df.columns:
        m = normalize_mode(df["mode"])
        if (m == "DISCH").any():
            df = df[m == "DISCH"].copy()

    df = df.sort_values("t_ms").reset_index(drop=True)

    # Sinais RAW
    t_ms = df["t_ms"].to_numpy(dtype=float)
    t_s = (t_ms - t_ms[0]) / 1000.0
    I = df["I_A"].to_numpy(dtype=float)
    V = df["Vbat_mean"].to_numpy(dtype=float)
    T = df["T_C"].to_numpy(dtype=float)

    # Diagnóstico
    dt = np.diff(t_s)
    dt_med = float(np.median(dt)) if len(dt) else np.nan
    di_raw = np.diff(I, prepend=I[0])
    max_di = float(np.max(np.abs(di_raw))) if len(di_raw) else 0.0

    print("Diagnóstico do StressDischarge (RAW):")
    print(f"- Amostras: {len(df)}")
    print(f"- dt mediano: {dt_med:.3f} s")
    print(f"- Corrente: min={I.min():.3f} A | max={I.max():.3f} A | range={(I.max()-I.min()):.3f} A")
    print(f"- Maior ΔI RAW entre linhas seguidas: {max_di:.4f} A")
    print(f"- Tensão: min={V.min():.3f} V | max={V.max():.3f} V")
    print(f"- Temperatura: min={T.min():.2f} °C | max={T.max():.2f} °C")

    # Parâmetros (ajustados para 1 Hz)
    di_threshold_raw = 0.8   # A: degrau mínimo (pegando degraus grandes e limpos)
    window_s = max(5.0, 5.0 * dt_med)  # 5 s é bom para 1 Hz
    min_points = 3           # com 1 Hz, janela 5 s dá ~5 pontos; 3 garante robustez

    print("\nParâmetros usados:")
    print(f"- di_threshold_raw: {di_threshold_raw:.2f} A")
    print(f"- window_s: {window_s:.2f} s")
    print(f"- min_points por janela: {min_points}")

    # Detecta índices de degrau por diferença RAW entre linhas consecutivas
    idx_candidates = np.where(np.abs(di_raw) >= di_threshold_raw)[0]

    events = []
    last_t0 = -1e9

    for idx in idx_candidates:
        t0 = float(t_s[idx])

        # Evita pegar a mesma transição repetida (deixa “espaço” de 2s)
        if (t0 - last_t0) < 2.0:
            continue
        last_t0 = t0

        before = (t_s >= (t0 - window_s)) & (t_s < t0)
        after  = (t_s >= t0) & (t_s <= (t0 + window_s))

        if before.sum() < min_points or after.sum() < min_points:
            continue

        I1 = float(np.mean(I[before]))
        V1 = float(np.mean(V[before]))
        T1 = float(np.mean(T[before]))

        I2 = float(np.mean(I[after]))
        V2 = float(np.mean(V[after]))
        T2 = float(np.mean(T[after]))

        dI = I2 - I1
        dV = V2 - V1
        if abs(dI) < 1e-9:
            continue

        R = -dV / dI
        T_evt = 0.5 * (T1 + T2)

        events.append({
            "t_event_s": t0,
            "i_before_a": I1,
            "i_after_a": I2,
            "v_before_v": V1,
            "v_after_v": V2,
            "T_before_C": T1,
            "T_after_C": T2,
            "dI_a": dI,
            "dV_v": dV,
            "R_ohm": float(R),
            "T_event_C": float(T_evt),
        })

    events_df = pd.DataFrame(events)

    if len(events_df) == 0:
        print("\nNão detectei eventos suficientes com esses parâmetros.")
        print("Ajustes que normalmente resolvem (nessa ordem):")
        print("1) reduzir di_threshold_raw para 0.6")
        print("2) aumentar window_s para 8.0 ou 10.0")
        return

    # Filtra R plausível
    events_df = events_df[np.isfinite(events_df["R_ohm"])].copy()
    events_df = events_df[(events_df["R_ohm"] > 0) & (events_df["R_ohm"] < 0.5)].reset_index(drop=True)

    if len(events_df) == 0:
        print("\nDetectei eventos, mas todos deram R fora do intervalo plausível (0..0.5 ohm).")
        print("Isso pode acontecer se o degrau não for “limpo” (OCV mudando muito na janela).")
        print("Ajuste: aumente di_threshold_raw (pegar só degraus grandes) e/ou reduza window_s.")
        return

    # Salva eventos
    out_events = RESULTS_TABLES_DIR / "stress_rT_events.csv"
    events_df.to_csv(out_events, index=False)

    # Ajusta R(T)
    params = fit_linear_rT(events_df)
    out_params = MODELS_DIR / "rT_params.csv"
    pd.DataFrame([params]).to_csv(out_params, index=False)

    print("\nOK ✅")
    print(f"- Eventos detectados: {params['n_events']}")
    print(f"- Faixa T nos eventos: {params['T_min_C']:.2f} a {params['T_max_C']:.2f} °C")
    print(f"- R mediana (eventos): {params['R_median_ohm']:.6f} ohm")
    print(f"- Modelo R(T): R_ref={params['R_ref_ohm']:.6f} ohm em T_ref={params['T_ref_C']:.2f} °C")
    print(f"           k={params['k_ohm_per_C']:.8f} ohm/°C | R²={params['R2']:.3f}")
    print(f"- Salvo em:")
    print(f"  {out_events}")
    print(f"  {out_params}")


if __name__ == "__main__":
    main()
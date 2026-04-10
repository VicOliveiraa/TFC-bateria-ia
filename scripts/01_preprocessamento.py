from __future__ import annotations

import pandas as pd
import numpy as np
from pathlib import Path

from config import (
    RAW_DIR,
    PROCESSED_DIR,
    RESULTS_TABLES_DIR,
    RAW_COLUMNS,
    MOVING_AVG_WINDOW,
    CLIP_CURRENT_A,
    CLIP_VOLTAGE_V,
    CLIP_TEMP_C,
)


def ensure_dirs():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)


def read_raw_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Verifica se todas as colunas necessárias existem
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"O arquivo {path.name} não contém as colunas esperadas: {missing}"
        )

    return df


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Ordena por tempo
    df = df.sort_values("t_ms").reset_index(drop=True)

    # Tempo em segundos
    df["time_s"] = (df["t_ms"] - df["t_ms"].iloc[0]) / 1000.0
    df["dt_s"] = df["time_s"].diff().fillna(0.0)

    # Renomeia colunas
    df = df.rename(
        columns={
            "Vbat_mean": "v_bat_v",
            "Vbus_INA": "v_bus_v",
            "Vbat_ADS": "v_ads_v",
            "Vsh_mV": "vsh_mv",
            "I_A": "i_a",
            "T_C": "t_c",
        }
    )

    # Clipping defensivo
    df["i_a"] = df["i_a"].clip(-CLIP_CURRENT_A, CLIP_CURRENT_A)
    df["v_bat_v"] = df["v_bat_v"].clip(*CLIP_VOLTAGE_V)
    df["v_ads_v"] = df["v_ads_v"].clip(*CLIP_VOLTAGE_V)
    df["v_bus_v"] = df["v_bus_v"].clip(*CLIP_VOLTAGE_V)
    df["t_c"] = df["t_c"].clip(*CLIP_TEMP_C)

    # Potência instantânea
    df["p_w"] = df["v_bat_v"] * df["i_a"]

    # Filtro média móvel simples
    if MOVING_AVG_WINDOW >= 2:
        for col in ["v_bat_v", "i_a", "t_c", "p_w"]:
            df[col] = df[col].rolling(
                window=MOVING_AVG_WINDOW, center=True, min_periods=1
            ).mean()

    return df


def summarize(df: pd.DataFrame, name: str) -> dict:
    duration_s = df["time_s"].iloc[-1]
    duration_h = duration_s / 3600.0

    ah_total = (df["i_a"] * df["dt_s"]).sum() / 3600.0
    wh_total = (df["p_w"] * df["dt_s"]).sum() / 3600.0

    return {
        "file": name,
        "duration_s": duration_s,
        "duration_h": duration_h,
        "mean_current_a": df["i_a"].mean(),
        "v_start_v": df["v_bat_v"].iloc[0],
        "v_end_v": df["v_bat_v"].iloc[-1],
        "temp_start_c": df["t_c"].iloc[0],
        "temp_end_c": df["t_c"].iloc[-1],
        "ah_total": ah_total,
        "wh_total": wh_total,
    }


def main():
    ensure_dirs()

    raw_files = sorted(RAW_DIR.glob("*.csv"))
    if not raw_files:
        print("Nenhum CSV encontrado em data/raw/")
        return

    summaries = []

    for file_path in raw_files:
        print(f"Processando {file_path.name}...")
        df_raw = read_raw_csv(file_path)
        df_proc = preprocess_dataframe(df_raw)

        out_path = PROCESSED_DIR / file_path.name
        df_proc.to_csv(out_path, index=False)

        summaries.append(summarize(df_proc, file_path.name))

    summary_df = pd.DataFrame(summaries)
    summary_path = RESULTS_TABLES_DIR / "summary_tests.csv"
    summary_df.to_csv(summary_path, index=False)

    print("\nProcessamento concluído.")
    print(f"Arquivos salvos em: {PROCESSED_DIR}")
    print(f"Resumo salvo em: {summary_path}")


if __name__ == "__main__":
    main()
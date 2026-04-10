import pandas as pd
from pathlib import Path

from config import (
    PROCESSED_DIR,
    DERIVED_DIR,
)

def ensure_dirs():
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

def process_discharge(file_path: Path):
    df = pd.read_csv(file_path)

    # Filtra somente descarga
    df_dis = df[df["mode"] == "DISCH"].copy()

    if len(df_dis) == 0:
        return None

    # Integra corrente para obter capacidade total (Ah)
    capacity_ah = (df_dis["i_a"] * df_dis["dt_s"]).sum() / 3600.0

    # Carga acumulada ao longo do tempo
    df_dis["ah_accum"] = (df_dis["i_a"] * df_dis["dt_s"]).cumsum() / 3600.0

    # SoC verdadeiro (1 no início → 0 no fim)
    df_dis["soc_gt"] = 1.0 - (df_dis["ah_accum"] / capacity_ah)

    # Limita entre 0 e 1 por segurança
    df_dis["soc_gt"] = df_dis["soc_gt"].clip(0.0, 1.0)

    return df_dis, capacity_ah

def main():
    ensure_dirs()

    processed_files = sorted(PROCESSED_DIR.glob("*.csv"))

    summary = []

    for file_path in processed_files:
        result = process_discharge(file_path)

        if result is None:
            continue

        df_dis, capacity_ah = result

        out_path = DERIVED_DIR / file_path.name
        df_dis.to_csv(out_path, index=False)

        summary.append({
            "file": file_path.name,
            "capacity_ah": capacity_ah
        })

        print(f"{file_path.name} → Capacidade: {capacity_ah:.4f} Ah")

    if summary:
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv(DERIVED_DIR / "capacity_summary.csv", index=False)
        print("\nResumo salvo em data/derived/capacity_summary.csv")

if __name__ == "__main__":
    main()
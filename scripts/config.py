from pathlib import Path

# Caminho raiz do projeto (um nível acima de /src)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Pastas principais
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
DERIVED_DIR = DATA_DIR / "derived"

MODELS_DIR = PROJECT_ROOT / "models"

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_TABLES_DIR = RESULTS_DIR / "tables"
RESULTS_FIGURES_DIR = RESULTS_DIR / "figures"

# Parâmetros de pré-processamento
MOVING_AVG_WINDOW = 5
CLIP_CURRENT_A = 10.0
CLIP_VOLTAGE_V = (0.0, 5.5)
CLIP_TEMP_C = (-20.0, 80.0)

# Colunas esperadas no RAW
RAW_COLUMNS = [
    "t_ms",
    "Vbat_mean",
    "Vbus_INA",
    "Vbat_ADS",
    "Vsh_mV",
    "I_A",
    "mode",
    "T_C",
]

C_NOM_AH = 3.30      # capacidade nominal adotada para o estimador
R0_OHM = 0.060       # resistência inicial (chute inicial). depois refinamos com o stress
# Modelo de resistência dependente de temperatura (estimado do stress)
USE_RT_MODEL = False
T_REF_C = 46.99
R_REF_OHM = 0.026896
K_OHM_PER_C = -0.00046367
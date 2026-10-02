"""Central configuration: paths, physical constants and experiment settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MODELS = ROOT / "models"

TELEMETRY_FILE = DATA_RAW / "telemetry.csv.gz"
HEALTH_FILE = DATA_RAW / "link_health.csv"
FEATURES_FILE = DATA_PROCESSED / "features.csv"
RESIDUALS_FILE = DATA_PROCESSED / "span_residuals.csv"

# ---- Network topology (mirrors the paper) ---------------------------------
N_SPANS = 19            # spans per end-to-end link
BIN_MINUTES = 2         # telemetry aggregation bin
N_ROUTES = 10           # fibre routes; each has two traffic directions
N_BINS = 300            # 300 bins x 20 directional links = 6000 samples

# ---- Optical physics -------------------------------------------------------
LAUNCH_POWER_DBM = 1.0      # per-channel launch power
AMP_SAT_POWER_DBM = 4.0     # amplifier output clamp
OSNR_CONST_DB = 58.0        # 10log10(h*nu*B_ref) term for 0.1 nm reference
NLI_ETA_DB = -38.0          # GN-model nonlinear coefficient (per span)
FIBER_CUT_LOSS_DB = 40.0    # expected extra loss when the fibre is cut
LOS_MARGIN_DB = 15.0        # receiver loses signal below launch - margin
RX_LOW_MARGIN_DB = 7.0      # Rx power alarm: below launch - margin is unhealthy
LOSS_MEAS_NOISE_DB = 0.12   # span-loss telemetry noise (2-min bin average)
GAIN_MEAS_NOISE_DB = 0.06   # amplifier-gain telemetry noise

# ---- Labelling -------------------------------------------------------------
OSNR_UNHEALTHY_DB = 17.0    # below this the link is marginal/failing (label 1)
OSNR_OUTAGE_DB = 13.0       # below this traffic is actually down

# ---- Experiment ------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.25            # 75 % train / 25 % test as in the paper
FEATURES = ["X1", "X2"]
TARGET = "y"

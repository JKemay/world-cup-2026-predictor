"""Project-wide configuration and identifiers discovered during the spike."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw" / "sportradar"

# Sportradar Soccer Extended — trial tier
BASE_URL = "https://api.sportradar.com/soccer-extended/trial/v4/en"
REQUEST_PAUSE = 1.3  # seconds between live API calls; trial keys are rate-limited

# Identifiers confirmed during the Phase 0 spike
WC_COMPETITION_ID = "sr:competition:16"  # FIFA World Cup
WC_SEASON_ID = "sr:season:101177"        # FIFA World Cup 2026

# Cutoff between the WC 2026 group stage and knockout stage, for the temporal
# out-of-sample backtest. Between the last group match (Algeria-Austria,
# 2026-06-28T02:00Z) and the first knockout match (South Africa-Canada,
# 2026-06-28T19:00Z).
KNOCKOUT_START = "2026-06-28T12:00:00+00:00"

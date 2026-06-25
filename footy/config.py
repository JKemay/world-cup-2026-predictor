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

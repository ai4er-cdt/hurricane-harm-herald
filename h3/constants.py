from __future__ import annotations


# Position of AOI
# TODO: round these values
HAITI_LON_MIN: float = -74.53520783870444
HAITI_LON_MAX: float = -68.31651208717756
HAITI_LAT_MIN: float = 17.48216032912591
HAITI_LAT_MAX: float = 20.130736297658242

TEXAS_TO_MAINE_LON_MIN: float = -98.05986540280205
TEXAS_TO_MAINE_LON_MAX: float = -75.39402341548211
TEXAS_TO_MAINE_LAT_MIN: float = 24.491674407495886
TEXAS_TO_MAINE_LAT_MAX: float = 36.75816955589585


DMG_CLASSES_DICT = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
    "un-classified": 4
}

from __future__ import annotations

RANDOM_STATE = 17

EARTH_RADIUS = 6371009  # metres

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

ALL_EF_FEATURES = {
	"weather": [
		"max_sust_wind", "shortest_distance_to_track", "max_sust_wind", "min_p",
		"r_ne_34", "r_se_34", "r_nw_34", "r_sw_34", "r_ne_50",
		"r_se_50", "r_nw_50", "r_sw_50", "r_ne_64", "r_se_64",
		"r_nw_64", "r_sw_64", "strength"
	],
	"soil": ["soil_density", "sand_content", "clay_content", "silt_content"],
	"storm_surge": ["storm_surge"],
	"dem": ["elevation", "slope", "aspect", "dis2coast"]}

ALL_FEATURES_TO_SCALE = [
	"max_sust_wind", "shortest_distance_to_track", "max_sust_wind", "min_p",
	"r_ne_34", "r_se_34", "r_nw_34", "r_sw_34", "r_ne_50",
	"r_se_50", "r_nw_50", "r_sw_50", "r_ne_64", "r_se_64",
	"r_nw_64", "r_sw_64", "strength",
	"soil_density", "sand_content", "clay_content", "silt_content",
	"storm_surge",
	"elevation", "slope", "aspect", "dis2coast"
]

RF_BEST_EF_FEATURES = {
	"weather": [
		"max_sust_wind", "shortest_distance_to_track", "min_p",
		"r_nw_34", "r_sw_34",
	],
	"soil": ["soil_density", "sand_content", "clay_content", "silt_content"],
	"storm_surge": ["storm_surge"],
	"dem": ["elevation", "slope", "aspect", "dis2coast"]}
RF_BEST_FEATURES_TO_SCALE = [
	"max_sust_wind", "shortest_distance_to_track", "min_p",
	"r_nw_34", "r_sw_34",
	"soil_density", "sand_content", "clay_content", "silt_content",
	"storm_surge",
	"elevation", "slope", "aspect", "dis2coast"
]

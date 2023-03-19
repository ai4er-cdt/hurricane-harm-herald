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
SHA1_xbd = {
	"xview2_geotiff.tgz": "6eae3baddf86796c15638682a6432e3e6223cb39",
	"xview2_geotiff.tgz.part-aa": "881ed94d1060c91e64c8eae438dfce492a21d9a9",
	"xview2_geotiff.tgz.part-ab": "4064dddc9aa05f786a3a6f70dd4ca86d79dd9e3a",
	"xview2_geotiff.tgz.part-ac": "0cfdf761e6f77ac5c423d9fb0927c3f8f8ac43da",
	"xview2_geotiff.tgz.part-ad": "44a39a7c4a80d386fb71ced95caee040126bb405",
	"xview2_geotiff.tgz.part-ae": "7fb96fac1d009b6a213d4efef2fbf5f1a475a554",
	"xview2_geotiff.tgz.part-af": "2ccbd04c4b2e27f8d948de734f661ec0c9d81152"
}

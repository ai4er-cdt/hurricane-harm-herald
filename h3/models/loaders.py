from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pickle
import torch

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from h3 import logger
from h3.models.balance_process import balance_process
from h3.utils.directories import get_metadata_pickle_dir, get_processed_data_dir, get_datasets_dir, get_pickle_dir
from h3.utils.dataframe_utils import read_and_merge_pkls, rename_and_drop_duplicated_cols


def get_df(balanced_data: bool) -> pd.DataFrame:
	data_dir = get_datasets_dir()
	if balanced_data:
		# This is the balanced_df
		logger.info("Loading balanced data")
		ECMWF_filtered_pickle_path = os.path.join(
			get_metadata_pickle_dir(),
			"filtered_lnglat_ECMWF_damage.pkl"
		)

		if os.path.exists(ECMWF_filtered_pickle_path):
			ECMWF_balanced_df = pd.read_pickle(ECMWF_filtered_pickle_path)
		else:
			ECMWF_balanced_df = balance_process(data_dir, "ECMWF")

		# remove unclassified class
		ECMWF_balanced_df = ECMWF_balanced_df[ECMWF_balanced_df.damage_class != 4]
		ECMWF_balanced_df["id"] = ECMWF_balanced_df.index
		# TODO: the ECMWF is not used

		# this does have the soil and terrain data in it
		filtered_pickle_path = os.path.join(
			get_metadata_pickle_dir(),
			"filtered_lnglat_pre_pol_post_damage.pkl"
		)
		if os.path.exists(filtered_pickle_path):
			df = pd.read_pickle(filtered_pickle_path)
		else:
			df = balance_process(data_dir)
	else:
		logger.info("Loading unbalanced data")
		# weather
		df_noaa_xbd_pkl_path = os.path.join(
			data_dir, "EFs/weather_data/xbd_obs_noaa_six_hourly_larger_dataset.pkl"
		)
		# terrain efs
		df_terrain_efs_path = os.path.join(
			get_processed_data_dir(),
			"Terrian_EFs.pkl"
		)
		# flood and soil properties
		df_topographic_efs_path = os.path.join(
			get_processed_data_dir(),
			"df_points_posthurr_flood_risk_storm_surge_soil_properties.pkl"
		)
		pkl_paths = [df_noaa_xbd_pkl_path, df_topographic_efs_path, df_terrain_efs_path]
		EF_df = read_and_merge_pkls(pkl_paths)
		df = rename_and_drop_duplicated_cols(EF_df)

	# remove unclassified, i.e. class damage == 4
	df = df[df.damage_class != 4]
	df["id"] = df.index

	return df


def train_val_test_df(
		df: pd.DataFrame,
		split_val_train_test: list,
		spatial: bool,
		hurricanes: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	train_test_value = split_val_train_test[2]
	train_val_value = split_val_train_test[1] / split_val_train_test[0]

	if spatial:
		train_event_names = hurricanes["train"]
		test_event_names = hurricanes["test"]
		train_df = df[df["disaster_name"].isin(train_event_names)]
		if train_event_names == test_event_names:
			# this won't solve the issue when only one hurricane matches
			# same training and test, spliting the dataset in to 70 20 10
			train_df, test_df = train_test_split(train_df, test_size=train_test_value, random_state=1)
		else:
			# otherwise, test is everything in df for test hurricanes
			# and the split is 70 20 10 with 10 left over
			test_df = df[df["disaster_name"].isin(test_event_names)]    # TODO: this problematic
			train_df, _ = train_test_split(train_df, test_size=train_test_value, random_state=1)
		train_df, val_df = train_test_split(train_df, test_size=train_val_value, random_state=1)
	else:
		train_df, test_df = train_test_split(df, test_size=train_test_value, random_state=1)
		train_df, val_df = train_test_split(train_df, test_size=train_val_value, random_state=1)

	return train_df, val_df, test_df


def get_class_weights(balanced_data: bool, train_df: pd.DataFrame) -> torch.Tensor | None:
	"""Return the class weights according if the data is balanced or unbalanced.
	If the data is unbalanced, returns None.

	Parameters
	----------
	balanced_data : bool
		True if the data is balanced.
	train_df : pd.DataFrame
		The train data to get the class weights (if balanced_data is True).

	Returns
	-------
	torch.Tensor or None
		If balanced data is False, returns None.
		Otherwise, returns a torch.Tensor of the class weights of the balanced data.
	"""
	if not balanced_data:
		class_weights = compute_class_weight(
			class_weight="balanced",
			classes=np.unique(train_df["damage_class"].to_numpy()),
			y=train_df["damage_class"]
		)
		class_weights = torch.as_tensor(class_weights).type(torch.FloatTensor)
	else:
		class_weights = None
	return class_weights


def scale_df(
		train_df: pd.DataFrame,
		val_df: pd.DataFrame,
		test_df: pd.DataFrame,
		features_to_scale: list[str],
		scaler,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	"""Scale the different dataframe according to a scaler.

	Parameters
	----------
	train_df : pd.DataFrame
		Train dataframe to scale.
		The scaler will be using this dataframe to fit.
	val_df : pd.DataFrame
		Validation dataframe to scale.
	test_df : pd.DataFrame
		Test dataframe to scale.
	features_to_scale : list of str
		The list of features to scale.
	scaler
		Scaler to use. Can take any from sklearn.preprocessing.

	Returns
	-------
	tuple of three pandas DataFrame.
		The three dataframe are the scaled train, scaled validation, and scaled test dataframes respectively.

	See Also
	--------
	sklearn.preprocessing
	"""
	scaled_train_df = train_df.copy()
	scaled_val_df = val_df.copy()
	scaled_test_df = test_df.copy()

	scaled_train_df[features_to_scale] = scaler.fit_transform(scaled_train_df[features_to_scale])
	scaled_val_df[features_to_scale] = scaler.transform(val_df[features_to_scale])
	scaled_test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
	return scaled_train_df, scaled_val_df, scaled_test_df


def save_test_df(
		balanced: bool,
		spatial: bool,
		hurricanes: dict[str, list[str]],
		split_val_train_test: list[float],
		features_to_scale: list,
		scaler
) -> None:
	"""Save test dataframe to pickle file.

	Parameters
	----------
	balanced : bool
		If True, uses balanced data.
	spatial : bool
		If True, uses spatial data, and using the hurricanes parameter.
	hurricanes : dict of list of string
		Dictionary with `train` `test` as key and taking a
		list of the hurricanes as value.
	split_val_train_test : list of float
		list of split for train validation and split, given as float.
		The values do not need to add up to 1.
	features_to_scale
	scaler :
		Scaler to use for the scaling data.
		See scale_df.
	"""
	df = get_df(balanced)
	# hurricanes = {
	# 	"test": ["MICHAEL", "MATTHEW"],
	# 	"train": ["MICHAEL", "MATTHEW"]
	# }
	train_df, val_df, test_df = train_val_test_df(
		df,
		split_val_train_test=split_val_train_test,
		spatial=False,
		hurricanes=hurricanes
	)
	b_string = "balanced" if balanced else "unbalanced"
	s_string = "spatial" if spatial else "non-spatial"
	splits = "".join(map(lambda x: str(int(x * 100)), split_val_train_test))
	name = f"test_df_{b_string}_{s_string}_{splits}.pickle"
	path = os.path.join(get_pickle_dir(), name)

	_, _, scaled_test_df = scale_df(train_df, val_df, test_df, features_to_scale, scaler)
	with open(path, "wb") as handle:
		pickle.dump(scaled_test_df, handle)
	logger.info(f"test_df saved at {path}")

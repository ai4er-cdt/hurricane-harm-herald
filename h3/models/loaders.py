from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from typing import Literal

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

from h3 import logger
from h3.dataloading.hurricane_dataset import HurricaneDataset
from h3.models.balance_process import balance_process
from h3.utils.directories import get_metadata_pickle_dir, get_processed_data_dir, get_datasets_dir
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
	# train_dataset = HurricaneDataset(
	# 	dataframe=scaled_train_df,
	# 	img_path=img_path,
	# 	EF_features=ef_features,
	# 	image_embedding_architecture=image_embedding_architecture,
	# 	zoom_levels=zoom_levels,
	# 	augmentations=augmentations,
	# 	ram_load=ram_load
	# )
	train_test_value = split_val_train_test[2]
	train_val_value = split_val_train_test[1] / split_val_train_test[0]

	if spatial:
		train_event_names = hurricanes["train"]
		test_event_names = hurricanes["test"]
		train_df = df[df["disaster_name"].isin(train_event_names)]
		test_df = df[df["disaster_name"].isin(test_event_names)]    # TODO: this problematic
		train_val_val_spatial = split_val_train_test[1] + split_val_train_test[2]
		train_df, val_df = train_test_split(train_df, test_size=train_val_val_spatial, random_state=1)
	else:
		train_df, test_df = train_test_split(df, test_size=train_test_value, random_state=1)
		train_df, val_df = train_test_split(train_df, test_size=train_val_value, random_state=1)

	return train_df, val_df, test_df


def get_class_weights(balanced_data: bool, train_df: pd.DataFrame) -> torch.Tensor | None:
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


def scale_df(train_df, val_df, test_df, features_to_scale, scaler) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
	scaled_train_df = train_df.copy()
	scaled_val_df = val_df.copy()
	scaled_test_df = test_df.copy()

	scaled_train_df[features_to_scale] = scaler.fit_transform(scaled_train_df[features_to_scale])
	scaled_val_df[features_to_scale] = scaler.transform(val_df[features_to_scale])
	scaled_test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])
	return scaled_train_df, scaled_val_df, scaled_test_df


def df_to_dataset(
		df: pd.DataFrame,
		img_path: str,
		ef_features: dict,
		image_embedding_architecture: Literal["ResNet18", "ViT_L_16", "Swin_V2_B", "SatMAE"],
		zoom_levels: list,
		ram_load: bool = False,
		augmentations=None
) -> HurricaneDataset:
	dataset = HurricaneDataset(
		dataframe=df,
		img_path=img_path,
		EF_features=ef_features,
		image_embedding_architecture=image_embedding_architecture,
		zoom_levels=zoom_levels,
		ram_load=ram_load,
		augmentations=augmentations
	)

	return dataset




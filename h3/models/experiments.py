from __future__ import annotations

import os
import warnings
import pandas as pd
# from google.colab import drive
import geopandas as gpd
import numpy as np
import pytorch_lightning as pl
import time
import torch

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

## Call function from basic models ipynb
from typing import List, Union
from pathlib import Path
from functools import reduce

from h3 import logger
from h3.dataprocessing.DataAugmentation import DataAugmentation
from h3.dataloading.HurricaneDataset import HurricaneDataset
from h3.models.multimodal import OverallModel
from h3.utils.directories import *
from h3.models.balance_process import main as balance_process_main

# from line_profiler_pycharm import profile


def check_files_in_list_exist(
		file_list: List[str] | List[Path]
):
	"""State which files don't exist and remove from list"""
	files_found = []
	for fl in file_list:
		# attempt conversion to Path object if necessary
		if type(fl) != Path:
			try:
				fl = Path(fl)
			except TypeError:
				print(f'{fl} could not be converted to Path object')

		if fl.is_file():
			files_found += fl,
		else:
			print(f'{fl} not found. Removing from list.')

	return files_found


def read_and_merge_pkls(
		pkl_paths: List[str] | List[Path]
) -> pd.DataFrame:
	"""Read in pkl files from list of file paths and merge on index"""
	# check all files exist
	pkl_paths_present = check_files_in_list_exist(pkl_paths)
	df_list = [pd.read_pickle(pkl) for pkl in pkl_paths_present]

	return reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True), df_list)


def rename_and_drop_duplicated_cols(
		df: pd.DataFrame
) -> pd.DataFrame:
	"""Drop columns which are copies of others and rename the 'asdf_x' headers which would have resulted"""
	# need to ensure no bad types first
	df = drop_cols_containing_lists(df)
	# remove duplicated columns
	dropped_df = df.T.drop_duplicates().T   # TODO: this a massive bottleneck
	# rename columns for clarity (especially those which are shared between dfs). Will be able to remove most with better
	# column naming further up the process
	new_col_names = {col: col.replace('_x', '') for col in dropped_df.columns if col.endswith('_x')}

	return dropped_df.rename(columns=new_col_names)


def drop_cols_containing_lists(
		df: pd.DataFrame
) -> pd.DataFrame:
	"""It seemed like the best solution at the time: and to be fair, I can't really think of better...
	N.B. for speed, only looks at values in first row – if there is a multi-type column, this would be the least of
	our worries...
	"""
	df = df.loc[:, df.iloc[0].apply(lambda x: type(x) != list)]

	return df


def main():
	torch.manual_seed(17)
	data_dir = get_datasets_dir()
	#
	# # weather
	# df_noaa_xbd_pkl_path = os.path.join(data_dir, 'weather/xbd_obs_noaa_six_hourly_larger_dataset.pkl')
	#
	# # terrain efs
	# df_terrain_efs_path = os.path.join(data_dir, "processed_data/Terrian_EFs.pkl")
	#
	# # flood and soil properties
	# df_topographic_efs_path = os.path.join(
	# 	data_dir,
	# 	"processed_data/df_points_posthurr_flood_risk_storm_surge_soil_properties.pkl")
	#
	# pkl_paths = [df_noaa_xbd_pkl_path, df_topographic_efs_path, df_terrain_efs_path]
	# EF_df = read_and_merge_pkls(pkl_paths)
	#
	# logger.info("rename and drop")
	# EF_df_no_dups = rename_and_drop_duplicated_cols(EF_df)
	# logger.info("done")
	#
	# # the below directory should be to the .pkl with all EFs
	# # img_path = "/content/images/"
	img_path = os.path.join(get_processed_data_dir(), "processed_xbd", "geotiffs_zoom", "images")
	#
	# EF_df_no_dups["id"] = EF_df_no_dups.index
	# EF_df_no_dups = EF_df_no_dups[EF_df_no_dups["damage_class"] != 4]
	# # df = EF_df_no_dups.sample(200) #sample rows for testing that it works
	# img_path = "/content/images/"

	ECMWF_filtered_pickle_path = os.path.join(
		data_dir,
		"processed_data/metadata_pickle/filtered_lnglat_ECMWF_damage.pkl"
	)

	if os.path.exists(ECMWF_filtered_pickle_path):
		ECMWF_balanced_df = pd.read_pickle(ECMWF_filtered_pickle_path)
	else:
		ECMWF_balanced_df = balance_process_main(data_dir, "ECMWF")

	# remove unclassified class
	ECMWF_balanced_df = ECMWF_balanced_df[ECMWF_balanced_df.damage_class != 4]
	ECMWF_balanced_df["id"] = ECMWF_balanced_df.index

	filtered_pickle_path = os.path.join(
		data_dir,
		"processed_data/metadata_pickle/filtered_lnglat_pre_pol_post_damage.pkl"
	)

	if os.path.exists(filtered_pickle_path):
		balanced_df = pd.read_pickle(filtered_pickle_path)
	else:
		balanced_df = balance_process_main(data_dir)

	# remove unclassified class
	balanced_df = balanced_df[balanced_df.damage_class != 4]
	balanced_df["id"] = balanced_df.index

	# EF_features = {
	# 	"weather": [
	# 		"max_sust_wind", "shortest_distance_to_track", "max_sust_wind", "min_p",
	# 		"r_ne_34", "r_se_34", "r_nw_34", "r_sw_34", "r_ne_50",
	# 		"r_se_50", "r_nw_50", "r_sw_50", "r_ne_64", "r_se_64",
	# 		"r_nw_64", "r_sw_64", "strength"
	# 	],
	# 	"soil": ["soil_density", "sand_content", "clay_content", "silt_content"],
	# 	"storm_surge": ["storm_surge"],
	# 	"dem": ["elevation", "slope", "aspect", "dis2coast"]}

	EF_features = {
		"weather": [
			"max_sust_wind", "shortest_distance_to_track", "min_p",
			"r_nw_34", "r_sw_34",
		],
		"soil": ["soil_density", "sand_content", "clay_content", "silt_content"],
		"storm_surge": ["storm_surge"],
		"dem": ["elevation", "slope", "aspect", "dis2coast"]}

	train_df, test_df = train_test_split(balanced_df, test_size=0.1, random_state=1)
	train_df, val_df = train_test_split(train_df, test_size=0.2 / 0.9, random_state=1)

	# features_to_scale = [
	# 	"max_sust_wind", "shortest_distance_to_track", "max_sust_wind", "min_p",
	# 	"r_ne_34", "r_se_34", "r_nw_34", "r_sw_34", "r_ne_50",
	# 	"r_se_50", "r_nw_50", "r_sw_50", "r_ne_64", "r_se_64",
	# 	"r_nw_64", "r_sw_64", "strength",
	# 	"soil_density", "sand_content", "clay_content", "silt_content",
	# 	"storm_surge",
	# 	"elevation", "slope", "aspect", "dis2coast"
	# ]

	features_to_scale = [
		"max_sust_wind", "shortest_distance_to_track", "min_p",
		"r_nw_34", "r_sw_34",
		"soil_density", "sand_content", "clay_content", "silt_content",
		"storm_surge",
		"elevation", "slope", "aspect", "dis2coast"
	]

	scaled_train_df = train_df.copy()
	scaled_val_df = val_df.copy()

	scaler = MinMaxScaler()
	scaled_train_df[features_to_scale] = scaler.fit_transform(scaled_train_df[features_to_scale])
	scaled_val_df[features_to_scale] = scaler.transform(val_df[features_to_scale])

	augmentations = DataAugmentation()
	# augmentations = None

	zoom_levels = ["1", "2", "4", "0.5"]
	zoom_levels = ["1"]
	# image_embedding_architecture = "ResNet18"
	image_embedding_architecture = "SatMAE"
	# image_embedding_architecture = "Swin_V2_B"

	cuda_device = torch.cuda.is_available()

	# class weights for weighted cross-entropy loss
	# class_weights = compute_class_weight(
	# 	class_weight="balanced",
	# 	classes=np.unique(train_df["damage_class"].to_numpy()),
	# 	y=train_df["damage_class"]
	# )

	# class_weights = torch.as_tensor(class_weights).type(torch.FloatTensor)
	ram_load = False

	train_dataset = HurricaneDataset(
		scaled_train_df, img_path, EF_features,
		image_embedding_architecture=image_embedding_architecture,
		zoom_levels=zoom_levels,
		augmentations=augmentations,
		ram_load=ram_load
	)

	val_dataset = HurricaneDataset(
		scaled_val_df, img_path, EF_features,
		image_embedding_architecture=image_embedding_architecture,
		zoom_levels=zoom_levels,
		ram_load=ram_load
	)
	if cuda_device:
		# torch.set_float32_matmul_precision('medium')
		num_workers = 4
		persistent_w = bool(num_workers)
	else:
		num_workers = 0
		persistent_w = False

	model = OverallModel(
		training_dataset=train_dataset,
		validation_dataset=val_dataset,
		num_input_channels=3,
		EF_features=EF_features,
		batch_size=64,
		image_embedding_architecture=image_embedding_architecture,
		image_encoder_lr=0,
		general_lr=1e-3,
		output_activation=None,
		loss_function_str="CELoss",
		num_output_classes=4,
		lr_scheduler_patience=3,
		zoom_levels=zoom_levels,
		# class_weights=class_weights,
		image_only_model=False,
		weight_decay=0.001,
		num_workers=num_workers,
		persistent_w=persistent_w
	)

	max_epochs = 30
	log_every_n_steps = 100

	early_stop_callback = EarlyStopping(monitor="val/loss", patience=5, mode="min")

	checkpoint_callback = ModelCheckpoint(
		monitor="val/loss",
		dirpath=os.path.join(
			get_checkpoint_dir(),
			f"{image_embedding_architecture}_{*zoom_levels,}_{ram_load}_balanced"
		),   # TODO: fix this path
		filename="{epoch}-{val/loss:.4f}",
		save_top_k=1,  # save the best model
		mode="min",
		every_n_epochs=1
	)

	tic = time.perf_counter()
	tensor_logger = TensorBoardLogger("tb_logs", name=f"{image_embedding_architecture}_{*zoom_levels,}")

	if cuda_device:
		logger.info("Setting the trainer using GPU")
		trainer = pl.Trainer(
			max_epochs=max_epochs,
			accelerator='gpu',
			log_every_n_steps=log_every_n_steps,
			callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar(refresh_rate=10)],
			logger=tensor_logger,
			precision="16-mixed"
		)
	else:
		logger.info("Setting the trainer using CPU")
		trainer = pl.Trainer(
			max_epochs=max_epochs,
			log_every_n_steps=log_every_n_steps,
			callbacks=[checkpoint_callback, early_stop_callback, TQDMProgressBar(refresh_rate=10)],
			logger=tensor_logger
		)

	# %reload_ext tensorboard
	# %tensorboard --logdir=lightning_logs/
	trainer.fit(model)

	toc = time.perf_counter()


if __name__ == "__main__":
	main()

from __future__ import annotations

import inspect
import os
import warnings
import pickle
import pandas as pd
# from google.colab import drive
import geopandas as gpd
import numpy as np
import pytorch_lightning as pl
import time
import torch
from tqdm.rich import tqdm

from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import TQDMProgressBar, RichProgressBar
from pytorch_lightning.loggers import TensorBoardLogger

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

## Call function from basic models ipynb
from typing import List, Literal, Callable
from pathlib import Path
from functools import reduce

from h3 import logger
from h3.dataprocessing.DataAugmentation import DataAugmentation
from h3.dataloading.HurricaneDataset import HurricaneDataset
from h3.models.multimodal import OverallModel
from h3.utils.directories import *
from h3.models.balance_process import main as balance_process_main
from h3.utils.simple_functions import rich_table

from h3.constants import RF_BEST_EF_FEATURES, RF_BEST_FEATURES_TO_SCALE
from h3.constants import ALL_EF_FEATURES, ALL_FEATURES_TO_SCALE
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
	dropped_df = df.T.drop_duplicates().T   # this a small bottleneck
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


def load_model_predict(
		path: str,
		train_dataset: pd.DataFrame,
		val_dataset: pd.DataFrame,
		test_df: pd.DataFrame,
		scaler,
		feature_to_scale,
		ef_features,
		img_path,
		augmentation,
		zooms,
		architecture
):
	name = os.listdir(os.path.join(path, "epoch=29-val"))
	model = OverallModel.load_from_checkpoint(
		os.path.join(path, "epoch=29-val", name[0]),
		training_dataset=train_dataset,
		validation_dataset=val_dataset
	)

	run_predict(
		model=model,
		test_df=test_df,
		pkl_name=os.path.basename(path),
		scaler=scaler,
		features_to_scale=feature_to_scale,
		ef_features=ef_features,
		img_path=img_path,
		augmentations=augmentation,
		zoom_levels=zooms,
		image_embedding_architecture=architecture,
	)


def run_predict(
		model,
		test_df: pd.DataFrame,
		pkl_name: str,
		scaler,
		features_to_scale: list[str],
		ef_features: dict[list[str]],
		img_path: str,
		augmentations,
		zoom_levels: list | None = None,
		image_embedding_architecture: Literal["ResNet18", "SatMAE", "Swin_V2_B"] = "ResNet18",
):
	model.eval()

	scaled_test_df = test_df.copy()
	test_save_path = os.path.join(get_pickle_dir(), 'test_df.pickle')
	with open(test_save_path, 'wb') as handle:
		pickle.dump(test_save_path, handle)

	scaled_test_df[features_to_scale] = scaler.transform(test_df[features_to_scale])

	test_dataset = HurricaneDataset(
		scaled_test_df,
		img_path,
		ef_features,
		image_embedding_architecture=image_embedding_architecture,
		zoom_levels=zoom_levels,
		augmentations=augmentations
	)
	predictions_list = []

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	if torch.cuda.is_available():
		model.to(device)

	with torch.no_grad():
		for index in tqdm(range(len(test_df)), desc="Eval model"):
			x, y = test_dataset[index]
			for key in x.keys():
				x[key] = x[key].unsqueeze(0).to(device)
			prediction = model(x)
			predictions_list.append(prediction)

	pickle_save_path = os.path.join(get_pickle_dir(), f'{pkl_name}_predi.pickle')
	logger.info("Pickling the saved data")
	with open(pickle_save_path, 'wb') as handle:
		pickle.dump(predictions_list, handle)


# please save the pickle file to google drive. it will be in hurricane-harm-herald/predictions_list_satmae.pickle
# also, manually save the tensorboard graphs, save as .csv


def run_model(
		ef_features: dict[list[str]] = RF_BEST_EF_FEATURES,
		features_to_scale: list[str] = RF_BEST_FEATURES_TO_SCALE,
		split_val_train_test: list | None = None,
		zoom_levels: list | None = None,
		balanced_data: bool = False,
		max_epochs: int = 30,
		log_every_n_steps: int = 100,
		image_embedding_architecture: Literal["ResNet18", "SatMAE", "Swin_V2_B"] = "ResNet18",
		use_augmentation: bool = True,
		ram_load: bool = False,
		num_worker: int = 4,
		precision: int | str = "16-mixed",
		torch_float32_precision: Literal["highest", "high", "medium"] = "highest",   # TODO: call it with medium
		predict: bool = True,
		ckp_name: str | None = None,
		spatial: bool = False,
		hurricanes: dict[str, list[str]] | None = None,
		load_only: bool = False
) -> None:

	cuda_device = torch.cuda.is_available()
	if ckp_name is None:
		ckp_name = f"{image_embedding_architecture}_{*zoom_levels,}_{'balance' if balanced_data else 'unbalanced'}"

	zoom_levels = zoom_levels or ["1"]
	split_val_train_test = split_val_train_test or [0.7, 0.2, 0.1]

	data_dir = get_datasets_dir()
	img_path = os.path.join(get_processed_data_dir(), "processed_xbd", "geotiffs_zoom", "images")

	train_test_value = split_val_train_test[2]
	train_val_value = split_val_train_test[1] / split_val_train_test[0]

	augmentations = DataAugmentation() if use_augmentation else None

	frame = inspect.currentframe()
	args, _, _, values = inspect.getargvalues(frame)
	rich_table(args, values, title="Model Parameters")
	logger.info(f"Cuda: {cuda_device}")

	if balanced_data:
		loss_function = "CELoss"
		ECMWF_filtered_pickle_path = os.path.join(
			get_metadata_pickle_dir(),
			"filtered_lnglat_ECMWF_damage.pkl"
		)

		if os.path.exists(ECMWF_filtered_pickle_path):
			ECMWF_balanced_df = pd.read_pickle(ECMWF_filtered_pickle_path)
		else:
			ECMWF_balanced_df = balance_process_main(data_dir, "ECMWF")

		# remove unclassified class
		ECMWF_balanced_df = ECMWF_balanced_df[ECMWF_balanced_df.damage_class != 4]
		ECMWF_balanced_df["id"] = ECMWF_balanced_df.index

		filtered_pickle_path = os.path.join(
			get_metadata_pickle_dir(),
			"filtered_lnglat_pre_pol_post_damage.pkl"
		)
		if os.path.exists(filtered_pickle_path):
			df = pd.read_pickle(filtered_pickle_path)
		else:
			df = balance_process_main(data_dir)
		# This is the balanced_df

	else:
		loss_function = "weighted_CELoss"
		# weather
		df_noaa_xbd_pkl_path = os.path.join(
			data_dir, 'EFs/weather_data/xbd_obs_noaa_six_hourly_larger_dataset.pkl'
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

	df = df[df.damage_class != 4]
	df["id"] = df.index

	if spatial:
		train_event_names = hurricanes["train"]
		test_event_names = hurricanes["test"]
		train_df = df[df["disaster_name"].isin(train_event_names)]
		test_df = df[df["disaster_name"].isin(test_event_names)]
		train_val_val_spatial = split_val_train_test[1] + split_val_train_test[2]
		train_df, val_df = train_test_split(train_df, test_size=train_val_val_spatial, random_state=1)
	else:
		train_df, test_df = train_test_split(df, test_size=train_test_value, random_state=1)
		train_df, val_df = train_test_split(train_df, test_size=train_val_value, random_state=1)

	if not balanced_data:
		class_weights = compute_class_weight(
			class_weight="balanced",
			classes=np.unique(train_df["damage_class"].to_numpy()),
			y=train_df["damage_class"]
		)
		class_weights = torch.as_tensor(class_weights).type(torch.FloatTensor)
	else:
		class_weights = None

	scaled_train_df = train_df.copy()
	scaled_val_df = val_df.copy()

	scaler = MinMaxScaler()
	scaled_train_df[features_to_scale] = scaler.fit_transform(scaled_train_df[features_to_scale])
	scaled_val_df[features_to_scale] = scaler.transform(val_df[features_to_scale])

	train_dataset = HurricaneDataset(
		dataframe=scaled_train_df,
		img_path=img_path,
		EF_features=ef_features,
		image_embedding_architecture=image_embedding_architecture,
		zoom_levels=zoom_levels,
		augmentations=augmentations,
		ram_load=ram_load
	)

	val_dataset = HurricaneDataset(
		dataframe=scaled_val_df,
		img_path=img_path,
		EF_features=ef_features,
		image_embedding_architecture=image_embedding_architecture,
		zoom_levels=zoom_levels,
		ram_load=ram_load
	)
	if cuda_device:
		torch.set_float32_matmul_precision(torch_float32_precision)
		num_workers = num_worker
		persistent_w = bool(num_workers)
	else:
		num_workers = 0
		persistent_w = False

	logger.info(f"Loss function is {loss_function}")
	logger.info(f"{num_workers} number of workers")

	if load_only:
		load_model_predict(
			path=os.path.join(
				get_checkpoint_dir(),
				ckp_name
			),
			train_dataset=train_df,
			val_dataset=val_df,
			test_df=test_df,
			scaler=scaler,
			feature_to_scale=features_to_scale,
			ef_features=ef_features,
			img_path=img_path,
			augmentation=augmentations,
			zooms=zoom_levels,
			architecture=image_embedding_architecture
		)
		return

	model = OverallModel(
		training_dataset=train_dataset,
		validation_dataset=val_dataset,
		num_input_channels=3,
		EF_features=ef_features,
		batch_size=64,
		image_embedding_architecture=image_embedding_architecture,
		image_encoder_lr=0,
		general_lr=1e-3,
		output_activation=None,
		loss_function_str=loss_function,
		num_output_classes=4,
		lr_scheduler_patience=3,
		zoom_levels=zoom_levels,
		class_weights=class_weights,
		image_only_model=False,
		weight_decay=0.001,
		num_workers=num_workers,
		persistent_w=persistent_w,
	)

	early_stop_callback = EarlyStopping(monitor="val/loss", patience=5, mode="min")

	checkpoint_callback = ModelCheckpoint(
		monitor="val/loss",
		dirpath=os.path.join(
			get_checkpoint_dir(),
			ckp_name
		),
		filename="{epoch}-{val/loss:.4f}",
		save_top_k=1,  # save the best model
		mode="min",
		every_n_epochs=1
	)

	tensor_logger = TensorBoardLogger("tb_logs", name=f"{image_embedding_architecture}_{*zoom_levels,}")

	if cuda_device:
		logger.info("Setting the trainer using GPU")
		trainer = pl.Trainer(
			max_epochs=max_epochs,
			accelerator='gpu',
			log_every_n_steps=log_every_n_steps,
			callbacks=[checkpoint_callback, early_stop_callback, RichProgressBar(refresh_rate=10)],
			logger=tensor_logger,
			precision=precision
		)
	else:
		logger.info("Setting the trainer using CPU")
		trainer = pl.Trainer(
			max_epochs=max_epochs,
			log_every_n_steps=log_every_n_steps,
			callbacks=[checkpoint_callback, early_stop_callback, TQDMProgressBar(refresh_rate=10)],
			logger=tensor_logger
		)

	trainer.fit(model)

	if predict:
		run_predict(
			model=model,
			test_df=test_df,
			scaler=scaler,
			features_to_scale=features_to_scale,
			ef_features=ef_features,
			img_path=img_path,
			augmentations=augmentations,
			zoom_levels=zoom_levels,
			image_embedding_architecture=image_embedding_architecture,
			pkl_name=ckp_name
		)


def main():
	zooms = ["1", "2", "4", "0.5"]
	zooms = ["1"]
	architecture: Literal["ResNet18", "SatMAE", "Swin_V2_B"]
	architecture = "ResNet18"
	spatial = True
	ckp_name = f"{architecture}_{*zooms,}_balanced_{spatial}"
	hurricanes = {
		"test": ["MICHAEL", "MATTHEW"],
		"train": ["MICHAEL", "MATTHEW"]
	}

	run_model(
		ef_features=ALL_EF_FEATURES,
		features_to_scale=ALL_FEATURES_TO_SCALE,
		split_val_train_test=[0.7, 0.2, 0.1],
		zoom_levels=zooms,
		balanced_data=False,
		max_epochs=30,
		log_every_n_steps=100,
		image_embedding_architecture=architecture,
		use_augmentation=True,
		ram_load=False,
		num_worker=4,
		precision="16-mixed",
		torch_float32_precision="medium",
		predict=True,
		ckp_name=ckp_name,
		spatial=True,
		hurricanes=hurricanes
	)


if __name__ == "__main__":
	main()

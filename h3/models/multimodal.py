from __future__ import annotations

import os
import random
from typing import Literal

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torchvision.models as models
import torchvision
import torch
import os
import torch.nn as nn
import torch.optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy
from PIL import Image
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torchvision.models import ViT_L_16_Weights, vit_l_16
from torchvision.models import swin_v2_b, Swin_V2_B_Weights
import random
import h3.models.SatMAE.utils

"""initialize the image embedding block"""
"""in GaLeNet Fig.1 this is the CLIP box"""


class ImageEncoder(pl.LightningModule):
	def __init__(self, image_embedding_architecture: Literal["ResNet18", "ViT_L_16", "Swin_V2_B", "SatMAE"]):
		super().__init__()

		if image_embedding_architecture == "ResNet18":
			# tell pytorch to use the ResNet18 architecture

			backbone = models.resnet18(weights="DEFAULT")

			# drop final layer since non-SSL trained model
			# with the below ResNet, num_image_encoder_features == 512
			layers = list(backbone.children())[:-1]
			self.feature_extractor = nn.Sequential(*layers)

		elif image_embedding_architecture == "ViT_L_16":
			# with the below ViT_L_16, num_image_encoder_features == 1024
			backbone = vit_l_16(weights=ViT_L_16_Weights.DEFAULT)
			layers = list(backbone.children())[:-1]
			self.feature_extractor = nn.Sequential(*layers)
			self.model = backbone

		elif image_embedding_architecture == "Swin_V2_B":
			# with the below Swin_V2_B, num_image_encoder_features == 1024
			backbone = swin_v2_b(weights=Swin_V2_B_Weights.DEFAULT)
			layers = list(backbone.children())[:-1]
			self.feature_extractor = nn.Sequential(*layers)

		elif image_embedding_architecture == "SatMAE":
			# with the below SatMAE, num_image_encoder_features == 1024
			self.feature_extractor = h3.models.SatMAE.utils.get_model()

		self.image_embedding_architecture = image_embedding_architecture

	def forward(self, x):
		self.feature_extractor.eval()

		with torch.no_grad():
			if self.image_embedding_architecture == "ResNet18":
				embedding = self.feature_extractor(x).flatten(1)

			elif self.image_embedding_architecture == "ViT_L_16":
				# the following code is taken from https://discuss.pytorch.org/t/feature-extraction-in-torchvision-models-vit-b-16/148029/2

				# This is the whole encoder sequence
				encoder = self.feature_extractor[1]

				# This is how the model preprocess the image.
				# The output shape is the one desired
				x = self.model._process_input(x)

				n = x.shape[0]

				batch_class_token = self.model.class_token.expand(n, -1, -1)
				x = torch.cat([batch_class_token, x], dim=1)
				x = encoder(x)

				# Classifier "token" as used by standard language architectures
				embedding = x[:, 0]

			elif self.image_embedding_architecture == "Swin_V2_B":
				embedding = self.feature_extractor(x)

			elif self.image_embedding_architecture == "SatMAE":
				embedding = self.feature_extractor(x)

		return embedding


"""initalize the generic encoder block"""
"""in GaLeNet Fig.1 this is all of the small grey Encoder Blocks"""


class GenericEncoder(pl.LightningModule):
	def __init__(self, num_input_features: int, num_output_features: int, dropout_rate: float):
		super().__init__()

		self.l1 = nn.Linear(num_input_features, num_output_features)
		self.batchnorm = nn.BatchNorm1d(num_output_features)
		self.dropout = nn.Dropout(dropout_rate)
		self.activation = nn.SiLU()

	def forward(self, x):
		x = self.l1(x)
		x = self.batchnorm(x)
		x = self.activation(x)
		x = self.dropout(x)

		return x


"""initalize the SoftMax classification layer to predict damage class"""


class ClassificationLayer(pl.LightningModule):
	def __init__(self, num_input_features: int, num_output_classes: int, output_activation: str | None):
		super().__init__()

		if output_activation == "sigmoid":  # use Sigmoid for binary classification
			self.activation = nn.Sigmoid()
			self.l1 = nn.Linear(num_input_features, 1)

		elif output_activation == "softmax":  # softmax for multiclass classification
			self.activation = torch.nn.Softmax(dim=1)
			self.l1 = self.l1 = nn.Linear(num_input_features, num_output_classes)

		elif output_activation == "relu":
			"""relu could be used if we treat damage classes as a regression
			problem. this is unbounded though so can produce values > 4."""
			self.activation = F.relu
			self.l1 = self.l1 = nn.Linear(num_input_features, 1)

		elif output_activation == None:
			self.activation = nn.Identity()
			self.l1 = self.l1 = nn.Linear(num_input_features, num_output_classes)

	def forward(self, x):
		x = self.l1(x)
		x = self.activation(x)

		return x


"""initalize the overall architecture, i.e. the combination of encoder blocks"""


class OverallModel(pl.LightningModule):
	"""
	Description of what this class does here

	Parameters
	----------
	training_dataset : torch.utils.data.Dataset
		Contains the data used for training

	validation_dataset : torch.utils.data.Dataset
		Contains the data used for training

	image_embedding_architecture : str
		Determines the image embedding architecture used. Possible values are:
			- 'ResNet18'
			- 'ViT_L_16'
			- 'Swin_V2_B'

	num_input_channels : int
		The number of channels in the input images.

	EF_features : dict(String: List(String))
		A dictionary mapping from type of EF to a list of strings of names of the EFs.
		E.g., {"weather": ["precip", "wind_speed"], "soil": ["clay", "sand"]}

	dropout_rate : float
		The dropout probability

	image_encoder_lr : float
		The learning rate for the image encoder. If 0, then image encoder weights are frozen.

	general_lr : float
		The learning rate for all other parts of the model.

	batch_size : int
		The batch size used during training and validation steps.

	weight_decay : float
		Adam weight decay (L2 penalty)

	lr_scheduler_patience : int
		The number of epochs of validation loss plateau before lr is decreased.

	num_image_feature_encoder_features : int
		The number of features output from the encoder that operates on the
		features produced by the image encoder

	num_output_classes : int
		The number of output classes. Set to 1 for regression.

	zoom_levels : List[str]
		A list containing the different image zoom levels.

	class_weights: torch.FloatTensor
		A tensor containing a weights to be applied to each class in the
		cross entropy loss function.

	image_only_model: Boolean
		If true, then the model behaves as if there were no EFs, and only the
		images are used to make predictions.

	loss_function_str : str
		Determines the loss function used. Possible values are:
			- 'BCELoss' : Binary Cross Entropy Loss, for binary classification
			- 'CELoss' : Cross Entropy Loss, for multiclass classification
			- 'MSE' : Mean Squared Error, for regression

	output_activation : str
		Determines the output activation function used. Possible values are:
			- 'sigmoid' : Sigmoid, for binary classification
			- 'softmax' : Softmax, for multiclass classification
			- 'relu' : ReLU, for regression


	Attributes
	----------
	Describe the attributes here, e.g. image_encoder, classification, augment
	"""

	def __init__(
			self,
			training_dataset,
			validation_dataset,
			image_embedding_architecture: Literal["ResNet18", "ViT_L_16", "Swin_V2_B", "SatMAE"] = "ResNet18",
			dropout_rate: float = 0.2,
			general_lr: float = 1e-4,
			image_encoder_lr: float = 0,
			batch_size: int = 32,
			weight_decay: float = 0.0,
			lr_scheduler_patience: int = 2,
			num_input_channels: int = 3,
			EF_features=None,
			num_concat_encoder_features: int = 100,
			num_image_feature_encoder_features: int = 56,
			num_output_classes: int = 4,
			zoom_levels: None | list = None,
			class_weights=None,
			image_only_model: bool = False,
			num_workers: int = 0,
			persistent_w: bool = False,
			loss_function_str: Literal["BCELoss", "CELoss", "MSE"] = "CELoss",  # maybe use focal loss for unbalanced multiclass as in GaLeNet
			output_activation:  Literal["sigmoid", "softmax", "relu"] | None = None  # CELoss expects unnormalized logits
	) -> None:
		super().__init__()

		if image_embedding_architecture == "ResNet18":
			num_image_encoder_features = 512
		else:  # every other case should be a ViT which outputs 1024 features
			num_image_encoder_features = 1024

		zoom_levels = ["1"] if zoom_levels is None else zoom_levels

		# total number of EFs present in the EF_features dictionary
		total_num_EFs = sum(map(len, EF_features.values()))

		# the image encoding architecture (e.g. ResNet)
		self.image_encoder = ImageEncoder(
			image_embedding_architecture
		)

		# need nn.ModuleList() to create a variable number of encoders depending
		# on the zoom levels supplied
		self.image_feature_encoders = nn.ModuleList()
		for _ in zoom_levels:
			# the encoding block for image features (produces Ai1 as in the diagram)
			self.image_feature_encoders.append(GenericEncoder(
				num_image_encoder_features, num_image_feature_encoder_features, dropout_rate
			))

		self.image_feature_classifiers = nn.ModuleList()
		for _ in zoom_levels:
			# the classification block for each embedded zoomed image
			self.image_feature_classifiers.append(ClassificationLayer(
				num_image_feature_encoder_features, num_output_classes, output_activation
			))

		if not image_only_model:
			# each EF specified in the EF_features dictionary gets a different
			# encoding block.

			self.ef_encoders = nn.ModuleDict()

			for key in EF_features:
				num_EFs = len(EF_features[key])  # num EFs in modality

				self.ef_encoders.update(
					{key: GenericEncoder(num_EFs, num_EFs, dropout_rate)}
				)

		if not image_only_model:
			# the encoder that takes as input the encoded image features + encoded EFs
			self.concat_encoder = GenericEncoder(
				(num_image_feature_encoder_features * len(zoom_levels)) + total_num_EFs, num_concat_encoder_features,
				dropout_rate
			)

		else:
			# the encoder that takes as input the encoded image features
			self.concat_encoder = GenericEncoder(
				(num_image_feature_encoder_features * len(zoom_levels)), num_concat_encoder_features, dropout_rate
			)

		# the classification layer used with the concatenated embedded features
		self.concat_classification = ClassificationLayer(
			num_concat_encoder_features,
			num_output_classes,
			output_activation
		)

		# """below, num_input_features should be some parameter that contains the number of weather related features"""
		# self.weather_encoder = GenericEncoder(
		#     num_input_features, num_output_features, dropout_rate
		# )

		# """below, num_input_features should be some parameter that contains the number of DEM related features"""
		# self.dem_encoder = GenericEncoder(
		#     num_input_features, num_output_features, dropout_rate
		# )

		# """there will be one storm surge feature. how, if at all, should this be encoded?"""
		# self.storm_surge_encoder = GenericEncoder(
		#     num_input_features, num_output_features, dropout_rate
		# )

		# """... more EF encoders"""

		if image_encoder_lr == 0:
			self.image_encoder.freeze()

		if loss_function_str == "BCELoss":
			self.loss_function = torch.nn.BCELoss()

		elif loss_function_str == "CELoss":
			self.loss_function = torch.nn.CrossEntropyLoss()

		elif loss_function_str == "MSE":
			self.loss_function = torch.nn.MSELoss()

		elif loss_function_str == "weighted_CELoss":
			self.loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

		self.image_encoder_lr = image_encoder_lr
		self.general_lr = general_lr
		self.batch_size = batch_size
		self.lr_scheduler_patience = lr_scheduler_patience
		self.zoom_levels = zoom_levels
		self.weight_decay = weight_decay
		self.image_only_model = image_only_model
		self.EF_features = EF_features
		self.num_workers = num_workers
		self.persistent_w = persistent_w

		self.training_dataset = training_dataset
		self.validation_dataset = validation_dataset

		# balanced accuracy (i think)
		self.accuracy = Accuracy(task='multiclass', average='macro', num_classes=num_output_classes)

		self.save_hyperparameters(ignore=["training_dataset", "validation_dataset"])

	def forward(self, inputs):
		"""for each zoom level Z, do image_Z_embedding = self.image_encoder(inputs["image_Z"], image_embedding_architecture)"""

		"""for each zoom level Z, do image_Z_embedding = self.GenericEncoder(image_Z_embedding, num_input_features, num_output_features)"""

		"""for each type of EF, do EF_embedding = self.image_encoder(inputs["EF"])"""
		"""all related EFs (e.g. all the weather EFs) should be in a
		single vector and pushed through a single embedding block."""
		"""there should be a different embedding block for each type of EF"""

		"""concat_embedding = concat all EF_embeddings and all image_Z_embeddings"""
		"""concat_embedding = GenericEncoder(concat_embedding)"""
		"""concat_prediction = self.ClassificationLayer(concat_prediction)"""

		"""for each zoom level Z, do Z_prediction = self.ClassificationLayer(image_Z_embedding)"""

		"""return the predictions from each zoom level individually and also the 
		predciction from the concat_embedding"""

		# return Z1_prediction, Z2_prediction, ..., concat_prediction

		# a list of tensors to be concatenated
		embeddings_to_concat = []

		# for each zoom level, put the image embedding tensor into the list
		for i in range(len(self.zoom_levels)):
			zoom_level = self.zoom_levels[i]
			image_zoom_embedding = self.image_encoder(inputs["img_zoom_" + zoom_level])
			image_zoom_embedding = self.image_feature_encoders[i](image_zoom_embedding)
			embeddings_to_concat.append(image_zoom_embedding)

		# a list of the predictions made from each embedded zoom level
		image_feature_predictions = []

		# for each embedded zoom level, predict the output class
		for i in range(len(embeddings_to_concat)):
			image_feature_predictions.append(self.image_feature_classifiers[i](embeddings_to_concat[i]))

		if not self.image_only_model:
			# put the embedded EFs into the the list
			for key in self.EF_features:
				embeddings_to_concat.append(self.ef_encoders[key](inputs[key]))

		# concats the EF and zoomed image embeddings. first dim is batch dimension, so concat along dim = 1
		concat_embedding = torch.concat(embeddings_to_concat, dim=1)
		concat_embedding = self.concat_encoder(concat_embedding)

		concat_predictions = self.concat_classification(concat_embedding)

		return concat_predictions, image_feature_predictions

	def _compute_losses(self, concat_predictions, image_feature_predictions, y):
		# def _compute_losses(self, Z1_prediction, Z2_prediction, ..., concat_prediction, y):

		"""Z1_loss = focal_loss(Z1_prediction, y)"""
		"""Z2_loss = focal_loss(Z2_prediction, y)"""
		"""..."""
		"""concat_loss = focal_loss(concat_loss, y)"""

		"""loss = sum(Z1_loss to Z4_loss) + concat_loss"""
		"""each of the individual loss functions is a torchvision.ops.focal_loss, as in GaLeNet"""
		"""the L_i's correspond to different zoom levels, so only use one L_i to start with"""

		# display(predictions.shape)
		# display(y.shape)
		# display(y.flatten().shape)

		loss = self.loss_function(concat_predictions, y.flatten())

		# as in GaLeNet, combine losses from concat embedding and each of the
		# zoomed image embeddings
		for image_feature_prediction in image_feature_predictions:
			loss += self.loss_function(image_feature_prediction, y.flatten())

		return loss

	def configure_optimizers(self):

		# this code allows different learning rates for the different blocks
		if not self.image_only_model:
			parameters = [
				{
					"params": self.image_encoder.parameters(),
					"lr": self.image_encoder_lr,
					"weight_decay": self.weight_decay,
				},
				{
					"params": self.concat_classification.parameters(),
					"lr": self.general_lr,
					"weight_decay": self.weight_decay,
				},
				{
					"params": self.concat_encoder.parameters(),
					"lr": self.general_lr,
					"weight_decay": self.weight_decay,
				},
				{
					"params": self.ef_encoders.parameters(),
					"lr": self.general_lr,
					"weight_decay": self.weight_decay,
				},
				{
					"params": self.image_feature_classifiers.parameters(),
					"lr": self.general_lr,
					"weight_decay": self.weight_decay,
				},
				{
					"params": self.image_feature_encoders.parameters(),
					"lr": self.general_lr,
					"weight_decay": self.weight_decay,
				},
			]
		else:
			# this else just removes ef_encoders
			parameters = [
				{
					"params": self.image_encoder.parameters(),
					"lr": self.image_encoder_lr,
					"weight_decay": self.weight_decay,
				},
				{
					"params": self.concat_classification.parameters(),
					"lr": self.general_lr,
					"weight_decay": self.weight_decay,
				},
				{
					"params": self.concat_encoder.parameters(),
					"lr": self.general_lr,
					"weight_decay": self.weight_decay,
				},
				{
					"params": self.image_feature_classifiers.parameters(),
					"lr": self.general_lr,
					"weight_decay": self.weight_decay,
				},
				{
					"params": self.image_feature_encoders.parameters(),
					"lr": self.general_lr,
					"weight_decay": self.weight_decay,
				},
			]

		optimizer = torch.optim.Adam(parameters)
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=30,
			eta_min=1e-4
		)
		return {
			"optimizer": optimizer,
			"lr_scheduler": lr_scheduler,
			"monitor": "val/loss",
		}

	def training_step(self, batch, *args, **kwargs):
		x, y = batch

		concat_predictions, image_feature_predictions = self.forward(x)
		loss = self._compute_losses(concat_predictions, image_feature_predictions, y).mean()

		acc = self.accuracy(concat_predictions, y)

		# belwo is for multizoom loss
		# Z1_prediction, Z2_prediction, ..., concat_prediction = self.forward(x)
		# loss = self._compute_losses(Z1_prediction, Z2_prediction, ..., concat_prediction, y).mean() # maybe normalize the loss?

		train_loss = self.all_gather(loss)  # what does all_gather do?
		self.log("train/loss", train_loss.mean(), logger=True, on_epoch=True)
		self.log("train accuracy", acc, logger=True, on_epoch=True)

		return train_loss

	def validation_step(self, batch, *args, **kwargs) -> Tensor | dict | list | tuple:
		x, y = batch

		concat_predictions, image_feature_predictions = self.forward(x)
		loss = self._compute_losses(concat_predictions, image_feature_predictions, y).mean()

		acc = self.accuracy(concat_predictions, y)

		# below code is for multi-zoom processing
		# Z05_prediction, Z1_prediction, Z2_prediction, ..., concat_prediction = self.forward(x)
		# loss = self._compute_losses(Z1_prediction, Z2_prediction, ..., concat_prediction, y).mean()

		val_loss = self.all_gather(loss)  # what does all_gather do?
		self.log("val/loss", val_loss.mean(), logger=True, on_epoch=True)
		self.log("val accuracy", acc, logger=True, on_epoch=True)
		return val_loss

	def train_dataloader(self) -> EVAL_DATALOADERS:
		loader = DataLoader(
			self.training_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			pin_memory=True,
			persistent_workers=self.persistent_w,
			shuffle=True,
		)
		return loader

	def val_dataloader(self) -> EVAL_DATALOADERS:
		loader = DataLoader(
			self.validation_dataset,
			batch_size=self.batch_size,
			num_workers=self.num_workers,
			pin_memory=True,
			persistent_workers=self.persistent_w,
		)
		return loader

import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics

from h3.constants import RANDOM_STATE
from h3.utils.directories import get_pickle_dir


def logistic_reg(x_train, y_train, x_test, y_test):
	model = LogisticRegression()
	model.fit(x_train, y_train)
	predictions = model.predict(x_test)
	model.score(x_test, y_test)
	importance = model.coef_[0]
	confusion_matrix = metrics.confusion_matrix(y_test, predictions)
	sns.heatmap(
		confusion_matrix / np.sum(confusion_matrix),
		annot=True,
		fmt=".2%",
		linewidths=.5,
		square=True,
		cmap="Blues_r"
	)
	plt.ylabel(r"Actual label")
	plt.xlabel(r"Predicted label")
	plt.show()


def main():

	# TODO: Rename this file to shorter name
	filename = "df_points_posthurr_flood_risk_storm_surge_soil_properties.pkl"
	filepath = os.path.join(get_pickle_dir(), filename)   # TODO: change this path

	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=RANDOM_STATE)
	logistic_reg(x_train, y_train, x_test, y_test)


if __name__ == "__main__":
	main()

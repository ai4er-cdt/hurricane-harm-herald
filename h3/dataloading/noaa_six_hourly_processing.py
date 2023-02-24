from __future__ import annotations

import pandas as pd
import numpy as np
import re


def convert_lat_lon(
	coord: str
) -> str:
	"""Convert lat/long of type 00N/S to +/-"""

	if 'S' in coord or 'W' in coord:
		val = '-' + (coord.translate({ord(i): '' for i in 'SW'})).strip()
		return val
	else:
		return coord.translate({ord(i): '' for i in 'NE'})


def preprocess_noaa_textfile(
	data: list
) -> list:
	"""Some data preprocessing before reading into pandas df.
	assigning event to each row, deleting headers, reformatting lat/long.
	Must have been read in from standard new NOAA .txt file format."""
	reformatted_data = []
	for i, line in enumerate(data):
		split_line = line.split(',')
		if re.search('[a-z]', split_line[0].lower()):
			line = ','.join([el.strip() for el in split_line])
			header = line
		else:
			split_line[4], split_line[5] = convert_lat_lon(
				split_line[4]), convert_lat_lon(split_line[5])
			reformatted_data.append(''.join((header, ','.join(split_line))))

	return reformatted_data


def reformat_noaa_df(
	df: pd.DataFrame
) -> pd.DataFrame:
	"""Tidy up data types in pd.DataFrame"""

	# convert columns to correct data type
	numeric_cols = df.columns.drop(
		['tag', 'name', 'date', 'time', 'record_id', 'sys_status'])
	df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
	# calculate storm intensity
	df['strength'] = df['max_sust_wind'].apply(windspeed_to_strength_category).astype('Int64')
	# combine date and time and correct format
	df['date'] = (df[['date', 'time']].agg(' '.join, axis=1)).apply(
		pd.to_datetime)
	# then drop time column
	df.drop('time', axis=1, inplace=True)
	# replace -999 values (shorthand for no data) with NaNs
	df.replace(-999, np.NaN, inplace=True)

	return df


def windspeed_to_strength_category(
	val: float | int
) -> bool:
	"""Assign an intensity value based on maximum sustained wind speed

	Parameters
	----------
		val : float | int
			numerical value to be compared
	Returns
	-------
		int
			storm categorisation
	"""
	wind_thresholds = [0, 64, 83, 96, 113, 137][::-1]
	cats = [0, 1, 2, 3, 4, 5][::-1]

	if np.isnan(val):
		return np.NaN
	else:
		for i, thresh in enumerate(wind_thresholds):
			if val >= thresh:
				return cats[i]


def return_most_recent_events_by_name(
	df: pd.DataFrame,
	event_names: list[str]
) -> pd.DataFrame:
	"""Returns the df containing the data for the most recent occurence of each
	event included in 'names'. df must have a 'date' column to judge most recent

	Parameters
	----------
	Returns
	-------
	restricted pd.DataFrame

	TODO: make this more flexible for selecting events
	"""
	# restrict to requested names only
	df_lim = df.loc[df['name'].isin(event_names)]
	# order df by date
	df_sorted = df_lim.sort_values(['name', 'date'], ascending=[True, False])
	# extract unique tags for most recent events
	recent_tags = df_sorted.groupby('name').first().tag

	return df_sorted.loc[df['tag'].isin(recent_tags)]
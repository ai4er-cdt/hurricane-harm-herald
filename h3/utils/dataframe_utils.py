from __future__ import annotations

import pandas as pd

from functools import reduce
from pathlib import Path

from h3.utils.simple_functions import check_files_in_list_exist


def read_and_merge_pkls(pkl_paths: list[str] | list[Path]) -> pd.DataFrame:
	"""Read in pkl files from list of file paths and merge on index.

	Parameters
	----------
	pkl_paths : list of str or list of pathlib.Path
		A list of paths to .pkl files to read and merge.

	Returns
	-------
	pandas.DataFrame
		A merged DataFrame containing all the data from the input .pkl files.

	See Also
	--------
	pd.DataFrame.merge()
	"""
	pkl_paths_present = check_files_in_list_exist(pkl_paths)
	df_list = [pd.read_pickle(pkl) for pkl in pkl_paths_present]
	return reduce(lambda df1, df2: pd.merge(df1, df2, left_index=True, right_index=True), df_list)


def drop_cols_containing_lists(df: pd.DataFrame) -> pd.DataFrame:
	"""Return a modified version of the input DataFrame with columns containing lists as values removed.

	Parameters:
	-----------
	df : pandas.DataFrame
		The DataFrame to modify.

	Returns:
	--------
	pandas.DataFrame
		A copy of the input DataFrame with columns containing lists removed.

	Notes:
	------
	This function looks only at the first row of the DataFrame and checks
	the type of each value to determine if it is a list.
	If a column contains any list values, it is dropped from the resulting DataFrame.
	This method is not necessarily the most efficient for dealing with multi-type columns.

	Examples
	--------
	>>> df = pd.DataFrame({'A': [1, 2, 3], 'B': ['foo', 'bar', 'baz'], 'C': [[1, 2], [3, 4], [5, 6]]})
	>>> df
		A    B       C
	0   1  foo  [1, 2]
	1   2  bar  [3, 4]
	2   3  baz  [5, 6]
	>>> drop_cols_containing_lists(df)
		A    B
	0   1  foo
	1   2  bar
	2   3  baz
	"""
	# It seemed like the best solution at the time: and to be fair, I can't really think of better...
	# N.B. for speed, only looks at values in first row – if there is a multi-type column, this would be the least of
	# our worries...
	df = df.loc[:, df.iloc[0].apply(lambda x: type(x) != list)]
	return df


def rename_and_drop_duplicated_cols(df: pd.DataFrame) -> pd.DataFrame:
	"""Drop columns which are copies of others and rename the `asdf_x` headers which would have resulted

	The function first checks for any bad types and removes any columns containing lists before removing any
	duplicated columns. The resulting DataFrame has its columns renamed for clarity, especially those which are shared
	between DataFrames.

	Parameters
	----------
	df : pandas.DataFrame
		The input DataFrame with possibly duplicated columns.

	Returns
	-------
	pandas.DataFrame
		A new DataFrame with duplicated columns removed and renamed.

	Examples
	--------
	>>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'C': [7, 8, 9], 'D': [7, 8, 9]})
	>>> df
		A  B  C  D
	0   1  4  7  7
	1   2  5  8  8
	2   3  6  9  9
	>>> rename_and_drop_duplicated_cols(df)
		A  B  C
	0   1  4  7
	1   2  5  8
	2   3  6  9
	"""
	# need to ensure no bad types first
	df = drop_cols_containing_lists(df)
	# remove duplicated columns
	dropped_df = df.T.drop_duplicates().T  # this a small bottleneck
	# rename columns for clarity (especially those which are shared between dfs). Will be able to remove most with better
	# column naming further up the process
	new_col_names = {col: col.replace("_x", "") for col in dropped_df.columns if col.endswith("_x")}

	return dropped_df.rename(columns=new_col_names)

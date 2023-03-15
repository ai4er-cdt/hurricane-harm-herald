from h3.utils.directories import get_data_dir
import pandas as pd
import os
from typing import List, Union
from pathlib import Path
from functools import reduce
from h3.constants import DMG_CLASSES_DICT


def check_files_in_list_exist(
    file_list: Union[List[str], List[Path]]
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
    pkl_paths: Union[List[str], List[Path]]
) -> pd.DataFrame:
    """Read in pkl files from list of file paths and merge on index"""
    # check all files exist
    pkl_paths_present = check_files_in_list_exist(pkl_paths)
    df_list = [pd.read_pickle(pkl) for pkl in pkl_paths_present]

    return reduce(lambda df1,df2: pd.merge(df1,df2,left_index=True,right_index=True), df_list)


def drop_cols_containing_lists(
    df: pd.DataFrame
) -> pd.DataFrame:
    """It seemed like the best solution at the time: and to be fair,
    I can't really think of better...
    N.B. for speed, only looks at values in first row – if there is a
    multi-type column, this would be the least of
    our worries...
    """
    df = df.loc[:, df.iloc[0].apply(lambda x: type(x) != list)]

    return df


def rename_and_drop_duplicated_cols(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Drop columns which are copies of others and rename the 'asdf_x'
    headers which would have resulted"""
    # need to ensure no bad types first
    df = drop_cols_containing_lists(df)
    # remove duplicated columns
    dropped_df = df.T.drop_duplicates().T
    # rename columns for clarity (especially those which are shared between dfs). Will be able to remove most with better
    # column naming further up the process
    new_col_names = {col: col.replace('_x', '') for col in dropped_df.columns if col.endswith('_x')}
    return dropped_df.rename(columns=new_col_names)


def data_loader():
    data_dir = get_data_dir()
    data_dir = "/Users/Lisanne/Documents/AI4ER/hurricane-harm-herald/data/test_folder"
    # ecmwf weather EFs
    df_ecmwf_xbd_pkl_path = os.path.join(data_dir,
                                         "EFs/weather_data/ecmwf/xbd_ecmwf_points.pkl")
    # NOAA weather EFs
    df_noaa_xbd_pkl_path = os.path.join(data_dir,
                                        "EFs/weather_data/xbd_obs_noaa_six_hourly_larger_dataset.pkl")
    # terrain efs
    df_terrain_efs_path = os.path.join(data_dir,
                                       "processed_data/Terrian_EFs.pkl")
    # flood, storm surge and soil properties
    df_topographic_efs_path = os.path.join(data_dir,
                                           "processed_data/df_points_posthurr_flood_risk_storm_surge_soil_properties.pkl")
    # distance to track, interpolated to different resolutions (ADD LATER)
    df_distance_to_track = os.path.join(data_dir,
                                        "processed_data/shortest_dis2hurricanes_varying_res.pkl")

    # based on feature importance
    all_pkl_paths = [df_noaa_xbd_pkl_path, df_terrain_efs_path
                     df_topographic_efs_path]
    # all_pkl_paths = [df_noaa_xbd_pkl_path, df_ecmwf_xbd_pkl_path, df_terrain_efs_path, df_topographic_efs_path,df_distance_to_track]

    all_EF_df = read_and_merge_pkls(all_pkl_paths)
    all_df_no_dups = rename_and_drop_duplicated_cols(all_EF_df)
    # drop r_max_wind as it is a column full of NaNs
    all_df_no_dups = all_df_no_dups.drop(columns=["r_max_wind"])

    map_dictionary = {v : k for k, v in DMG_CLASSES_DICT.items()}
    all_df_no_dups["damage_categorical"] = all_df_no_dups["damage_class"].replace(map_dictionary)
    return all_df_no_dups


def main():
    bperf_EF_df_no_dups = data_loader()

    n_sampled_dfs = []
    for damage_type in bperf_EF_df_no_dups.damage_class.unique():
        filtered_damage_df = bperf_EF_df_no_dups[bperf_EF_df_no_dups["damage_class"] == (damage_type)]
        value_counts = bperf_EF_df_no_dups.damage_class.value_counts().rename_axis('damage_class').reset_index(name='value_count')
        # damage class 3 is destroyed and is the baseline
        class3_value_count = int(value_counts[(value_counts['damage_class'] == 3)]["value_count"])
        if len(filtered_damage_df) >= class3_value_count:
            random_n_df = filtered_damage_df.sample(n=class3_value_count)
        else:
            random_n_df = filtered_damage_df
        n_sampled_dfs.append(random_n_df)
    balanced_df = pd.concat(n_sampled_dfs)
    return balanced_df


if __name__ == '__main__':
    main()

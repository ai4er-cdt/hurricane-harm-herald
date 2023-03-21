from __future__ import annotations

from typing import Tuple, Dict, Any, List

import cdsapi
import glob
import os
import urllib.request
import re

import geopy
import numpy as np
import pandas as pd
import xarray as xr

from shapely.geometry.point import Point
from tqdm import tqdm

from h3.dataloading import general_df_utils
from h3.dataprocessing import extract_metadata
from h3.utils import directories
from h3.utils.simple_functions import pad_number_with_zeros
from h3.utils.file_ops import guarantee_existence


def find_fetch_closest_station_files(
    df_xbd_points: pd.DataFrame,
    df_noaa: pd.DataFrame,
    df_stations: pd.DataFrame,
    download_dest_dir: str,
    time_buffer: tuple[float, str] = (1, "d"),
    min_number: int = 1,
    distance_buffer: float = None,
) -> pd.DataFrame:
    """Fetch csvs corresponding to the closest weather stations from each xBD data point.

    Parameters
    ----------
    df_xbd_points : pd.DataFrame
        pd.DataFrame of xBD points
    df_noaa : pd.DataFrame
        pd.DataFrame of NOAA Best Track data
    df_stations : pd.DataFrame
        pd.DataFrame of weather station metadata
    time_buffer : Tuple[float, str]
        period of time to require weather stations be active, either side of event activity
    download_dest_dir : str
        path to directory in which downloaded files are placed
    min_number : int default is 1
        minimum number of weather stations to return (in order of proximity to xBD datapoint)
    distance_buffer : float default is None
        distance by which to restrict the stations search. Defaults to None to expand search iteratively if no stations
        are found

    Returns
    -------
    pd.DataFrame
        xBD dataframe with additional information about the 'min_number' closest weather stations: their names and the
        start/end of the weather event
    """

    # pre-assign column of values for assignment
    df_xbd_points[["event_start", "event_end"]] = np.nan
    df_xbd_points[["closest_stations", "stations_lat_lons"]] = np.nan

    # group by event in df_xbd_points
    df_xbd_points_grouped = df_xbd_points.groupby("disaster_name")
    # for each group in df_xbd_points:
    for name, group in df_xbd_points_grouped:
        # calculate start and end of event
        # df_event_weather = df_noaa[df_noaa["name"] == name]
        start, end = calculate_first_last_dates_from_df(group, time_buffer)
        # limit stations df to those operational within +/- 1 time_buffer either side of event
        df_station_time_lim = df_stations[
            (df_stations["begin"] <= start) & (df_stations["end"] >= end)]

        ignore_csvs = []
        # for each xbd observation in group
        for index, obs in tqdm(group.iterrows(), total=len(group)):
            # limit stations spatially
            obs_lat_lons = [obs["lat"], obs["lon"]]
            df_station_spatial_time_lim = general_df_utils.limit_df_spatial_range(
                df_station_time_lim, obs_lat_lons, min_number, distance_buffer)

            stations_list = []
            station_no = 0
            while len(stations_list) < min_number:
                # find the closest weather station(s) to current weather station (allow closest N, or within limit)
                try:
                    station_index = general_df_utils.find_index_closest_point_in_col(
                        group["geometry"].loc[index], df_station_spatial_time_lim, "geometry", which_closest=station_no)
                    # get weather station csv filename
                    csv_filename = df_station_spatial_time_lim["csv_filenames"].loc[station_index]
                except: # noqa
                    df_station_spatial_time_lim = general_df_utils.limit_df_spatial_range(
                        df_station_time_lim, obs_lat_lons, len(df_station_spatial_time_lim)+1)
                    station_index = general_df_utils.find_index_closest_point_in_col(
                        group["geometry"].loc[index], df_station_spatial_time_lim, "geometry", which_closest=station_no)
                    # get weather station csv filename
                    csv_filename = df_station_spatial_time_lim["csv_filenames"].loc[station_index]

                event_year = start.year
                url = generate_station_url(event_year, csv_filename)

                # executes if weather station not already downloaded; if file in ignore, reloop to next-closest station
                if not "/".join((str(event_year), csv_filename)) in ignore_csvs:
                    # if file doesn't exist, append to ignore and reloop
                    # if file not downloaded
                    if not check_is_file_downloaded(csv_filename, download_dest_dir):
                        try:
                            download_dest = download_dest_dir + ".".join((csv_filename, "csv"))
                            urllib.request.urlretrieve(url, download_dest)
                            stations_list.append(csv_filename)
                        except: # noqa
                            ignore_csvs.append("/".join((str(event_year), csv_filename)))
                    else:
                        stations_list.append(csv_filename)
                station_no += 1

            # append list of stations
            df_xbd_points["closest_stations"].iloc[index] = stations_list
            # append start and end dates
            df_xbd_points["event_start"].iloc[index] = start
            df_xbd_points["event_end"].iloc[index] = end

        # remove station rows which don"t exist
        df_stations = df_stations.loc[~df_stations["csv_filenames"].isin(ignore_csvs)]

    return df_xbd_points


def generate_station_url(
    event_year: str,
    csv_filename: str
) -> str:
    """Generate weather station metadata .csv file URL"""
    URL_START = "https://www.ncei.noaa.gov/data/global-hourly/access/"
    return URL_START + "/".join((str(event_year), csv_filename)) + ".csv"


def check_is_file_downloaded(
    csv_filename: str,
    download_dest_dir: str
) -> bool:
    """True if already downloaded, False if not"""
    potential_file_path = "/".join((download_dest_dir, csv_filename)) + ".csv"
    if os.path.exists(potential_file_path):
        # downloaded
        return True
    else:
        print(f"{csv_filename} already downloaded.")
        return False


def generate_weather_stations_df(
    stations_meta_csv_file_path: str,
    lat_lon_range: list[list[float]] = [[0, 40], [-110, -50]]
):
    """Generate a df of valid weather stations within lat_lon_range of interest. Requires a path to the metadata file,
    which is found in the H3 GitHub repository at /h3/data_files/isd-metadata.csv
    This file is sourced from NOAA TODO: permalink

    Parameters
    ----------
    stations_meta_csv_file_path : str
        path to historic weather stations metadata
    lat_lon_range : list[float] defaults to [0, 40, -110, -50]
    """
    # load df with dates specified
    df_stations_all = general_df_utils.standardise_df(
        pd.read_csv(stations_meta_csv_file_path, parse_dates=["BEGIN", "END"]))
    # remove stations with key data missing
    df_stations_valid = df_stations_all.dropna(subset=["lat", "lon", "usaf", "wban"])
    # limit stations to latitude-longitude range of interest
    df_stations_lim = general_df_utils.exclude_df_rows_by_range(
        df_stations_valid, ["lat", "lon"], [lat_lon_range[0], lat_lon_range[1]])
    # generating filename of hourly weather data
    df_stations = general_df_utils.concat_df_cols(df_stations_lim, "csv_filenames", ["usaf", "wban"])

    return df_stations


def generate_and_save_noaa_best_track_pkl(
    noaa_meta_txt_file_path: str,
    xbd_hurricanes_only: bool = False
) -> pd.DataFrame:
    """Wrapper for generate_noaa_best_track_pkl. Generates a pandas DataFrame from a NOAA best track text file. Then 
    saves this to the correct directory: data/dataset/weather/noaa with the correct filename.

    The function takes in a file path to a NOAA best track text file, and reads in the data.
    It then preprocesses the data, reformats it into a pandas DataFrame, and returns the DataFrame.

    Parameters
    ----------
    noaa_meta_txt_file_path : str
        File path to the NOAA best track text file. This can be found in the H3 GitHub repository at
        /h3/data_files/hurdat2-1851-2021-meta.txt

    Returns
    -------
    None
    """
    df = generate_noaa_best_track_pkl(noaa_meta_txt_file_path, xbd_hurricanes_only)
    noaa_data_dir = directories.get_noaa_data_dir()
    if xbd_hurricanes_only:
        file_name = 'noaa_xbd_hurricanes.pkl'
    else:
        file_name = 'noaa_hurricanes.pkl'
        
    save_pkl_to_structured_dir(df, file_name)

    return df
        

def generate_noaa_best_track_pkl(
    noaa_meta_txt_file_path: str,
    xbd_hurricanes_only: bool = False
):
    """Generates a pandas DataFrame from a NOAA best track text file, and returns it.

    The function takes in a file path to a NOAA best track text file, and reads in the data.
    It then preprocesses the data, reformats it into a pandas DataFrame, and returns the DataFrame.

    Parameters
    ----------
    noaa_meta_txt_file_path : str
        File path to the NOAA best track text file. This can be found in the H3 GitHub repository at
        /h3/data_files/hurdat2-1851-2021-meta.txt

    Returns
    -------
    pd.DataFrame
        pd.DataFrame containing the reformatted NOAA best track data.
    """
    with open(noaa_meta_txt_file_path, "r") as noaa_txt_file:
        noaa_data = noaa_txt_file.read().strip().split("\n")

    reformatted_noaa_data = preprocess_noaa_textfile(noaa_data)

    noaa_longform_col_names = [
        "tag", "name", "num_entries", "date", "time", "record_id", "sys_status",  "lat", "long", "max_sust_wind",
        "min_p", "r_ne_34", "r_se_34", "r_nw_34", "r_sw_34", "r_ne_50", "r_se_50", "r_nw_50", "r_sw_50",
        "r_ne_64", "r_se_64", "r_nw_64", "r_sw_64", "r_max_wind"
    ]

    noaa_df = reformat_noaa_df(
        pd.DataFrame([el.split(",") for el in reformatted_noaa_data], columns=noaa_longform_col_names))

    if xbd_hurricanes_only:
        # Restrict NOAA data to xbd events
        xbd_hurricane_names = ["MICHAEL", "MATTHEW", "FLORENCE", "HARVEY"]
        noaa_df = return_most_recent_events_by_name(noaa_df, xbd_hurricane_names)

    return noaa_df


def find_first_timestamp_index(lst):
    """
    Given a list of type objects, return the first index where the type is either pd.Timestamp or datetime64_any_dtype.
    """
    for i, item in enumerate(lst):
        if item == pd.Timestamp or np.issubdtype(item, np.datetime64):
            return i


def preprocess_noaa_textfile(
    data: list
) -> list:
    """Some data preprocessing before reading into pandas df.
    assigning event to each row, deleting headers, reformatting lat/long.
    Must have been read in from standard new NOAA .txt file format."""
    reformatted_data = []
    for i, line in enumerate(data):
        split_line = line.split(",")
        if re.search("[a-z]", split_line[0].lower()):
            line = ",".join([el.strip() for el in split_line])
            header = line
        else:
            split_line[4] = convert_lat_lon(split_line[4])
            split_line[5] = convert_lat_lon(split_line[5])
            reformatted_data.append("".join((header, ",".join(split_line))))

    return reformatted_data


def reformat_noaa_df(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Tidy up data types in pd.DataFrame"""

    # convert columns to correct data type
    numeric_cols = df.columns.drop(
        ["tag", "name", "date", "time", "record_id", "sys_status"])
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
    # calculate storm intensity
    df["strength"] = df["max_sust_wind"].apply(windspeed_to_strength_category).astype("Int64")
    # combine date and time and correct format
    df["date"] = (df[["date", "time"]].agg(" ".join, axis=1)).apply(
        pd.to_datetime)
    # then drop time column
    df.drop("time", axis=1, inplace=True)
    # replace -999 values (shorthand for no data) with NaNs
    df.replace(-999, np.NaN, inplace=True)

    return df


def convert_lat_lon(
    coord: str
) -> str:
    """Convert lat/long of type 00N/S to +/-"""

    if "S" in coord or "W" in coord:
        val = "-" + (coord.translate({ord(i): "" for i in "SW"})).strip()
        return val
    else:
        return coord.translate({ord(i): "" for i in "NE"})


# checkThreshold
def windspeed_to_strength_category(
    val: float | int
) -> bool | int:
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


def calculate_first_last_dates_from_df(
    df: pd.DataFrame,
    time_buffer: tuple[float, str] = (0, "h"),
    date_col_name: list[str] = None
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Calculate the first and last dates from a df, with a time buffer before
    and after.

    Parameters
    ----------
    df : pd.DataFrame
        should contain at least one datetime column. If multiple datetime
        columns, column to be used should be specified. Will default to first
        occurence of such a column
    time_buffer : tuple[float,str] defaults to [0,'h'] (no buffer)
        extra time to remove from first occurence and add to last occurence.
        A string specifying the unit of time is necessary e.g. 'h' for
        hours (either as a tuple or list). See
        https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html for
        more info.
    date_col_name : list[str] defaults to None
        name of column containing datetime objects to be processed

    Returns
    -------
    tuple[pd.Timestamp]
        detailing start and end time/date

    N.B. currently ignoring any timezone information
    """
    # if no date_column_name provided
    if not date_col_name:
        try:
            # try to find first occurence of a coolumn containing dates
            col_types = [type(df[col].iloc[0]) for col in df.columns]
            for i, c in enumerate(col_types):
                if c == pd.Timestamp:
                    col_name = df.columns[i]
                    df[col_name] = df[col_name].astype("datetime64[ns]")
            # index = find_first_timestamp_index(col_types)
            # date_col = df.columns.tolist()[index]
            date_col = df.columns[df.apply(
                pd.api.types.is_datetime64_any_dtype)].tolist()[0]
        except TypeError():
            print("No column containing datetime64 objects found")
    else:
        # check if column provided contains datetime objects
        if pd.api.types.is_datetime64_any_dtype(df[date_col_name]):
            date_col = df[date_col_name]
        else:
            raise ValueError(
                "Column provided as date_col_name does not contain datetime objects")

    # if date column type has a timezone detailed
    if pd.api.types.is_datetime64tz_dtype(df[date_col]):
        # convert the "dates" column to naive timestamps
        df[date_col] = df[date_col].dt.tz_localize(None)

    # generate time buffer
    delta = pd.Timedelta(time_buffer[0], time_buffer[1])

    # find minimum and maximum date values
    start = df[date_col].min() - delta
    end = df[date_col].max() + delta

    return start, end


def return_most_recent_events_by_name(df: pd.DataFrame, event_names: list[str]) -> pd.DataFrame:
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
    df_lim = df.loc[df["name"].isin(event_names)]
    # order df by date
    df_sorted = df_lim.sort_values(["name", "date"], ascending=[True, False])
    # extract unique tags for most recent events
    recent_tags = df_sorted.groupby("name").first().tag

    return df_sorted.loc[df["tag"].isin(recent_tags)]


def download_ecmwf_files(download_dest_dir: str, distance_buffer: float = 5):
    """Load in ecmwf grib files from online"""

    # if file doesn"t exist at correct directory, generate it
    if os.path.exists(os.path.join(directories.get_noaa_data_dir(), "noaa_xbd_hurricanes.pkl")):
        df_noaa_xbd_hurricanes = pd.read_pickle(
            os.path.join(directories.get_noaa_data_dir(), "noaa_xbd_hurricanes.pkl"))
    else:
        df_noaa_xbd_hurricanes = generate_noaa_best_track_pkl(
            os.path.join(directories.get_h3_data_files_dir(), "hurdat2-1851-2021-meta.txt"), xbd_hurricanes_only=True)
        # save noaa pkl
        save_pkl_to_structured_dir(df_noaa_xbd_hurricanes, )

    if os.path.exists(os.path.join(directories.get_xbd_dir(), "xbd_data_points.pkl")):
        df_xbd_points = pd.read_pickle(os.path.join(directories.get_xbd_dir(), "xbd_data_points.pkl"))
    else:
        _, df_xbd_points = extract_metadata.main()
        save_pkl_to_structured_dir(df_xbd_points, "df_xbd_points.pkl")

    event_api_info, start_end_dates, areas = return_relevant_event_info(
        df_xbd_points,
        df_noaa_xbd_hurricanes,
        distance_buffer=distance_buffer,
        verbose=True)
    # these are the weather parameters with appropriate formats for download
    weather_keys = ["d2m", "t2m", "tp", "sp", "slhf", "e", "pev", "ro", "ssro", "sro", "u10", "v10"]
    weather_params = return_full_weather_param_strings(weather_keys)

    # call api to download ecmwf weather files
    fetch_era5_data(
        weather_params=weather_params,
        start_end_dates=start_end_dates,
        areas=areas,
        download_dest_dir=download_dest_dir)

    return df_xbd_points, df_noaa_xbd_hurricanes, weather_keys


def generate_and_save_era5_pkl(
    distance_buffer: float = 5
) -> pd.DataFrame:
    """Wrapper for generate_era5_pkl which also saves the output df to the correct folder location
    """
    df = generate_ecmwf_pkl(distance_buffer)

    save_pkl_to_structured_dir(df, 'era5_xbd_values.pkl')
    return df


def generate_ecmwf_pkl(
    distance_buffer: float = 5
) -> pd.DataFrame:
    download_dest_dir = directories.get_ecmwf_data_dir()
    df_xbd_points, df_noaa_xbd_hurricanes, weather_keys = download_ecmwf_files(download_dest_dir, distance_buffer)
    # download ecmwf grib files to separate directories within /datasets/weather/ecmwf/
    # group ecmwf xarray files into dictionary indexed by name of weather event
    xbd_event_xa_dict = generate_xbd_event_xa_dict(download_dest_dir, df_noaa_xbd_hurricanes)
    # generate df for all xbd points' closest maximum era5 values
    df_ecmwf_xbd_points = determine_ecmwf_values_from_points_df(
        xbd_event_xa_dict,
        weather_keys=weather_keys,
        df_points=df_xbd_points,
        )

    return df_ecmwf_xbd_points


# def merge_grib_files_to_nc(
#     grib_dir_path: str
# ):
#     """Merge all GRIB files in a directory into a single NetCDF file.

#     Parameters
#     ----------
#     grib_dir_path : str
#         Path to the directory containing the GRIB files.

#     Returns
#     -------
#     None
#     """
#     # load in all files in folder
#     file_paths = "/".join((grib_dir_path, "*.grib"))
#     grib_dir_name = "/".split(grib_dir_path)[-2]

#     xa_dict = {}
#     for file_path in tqdm.tqdm(glob.glob(file_paths)):
#         # get name of file
#         file_name = file_path.split("/")[-1]
#         # read into xarray
#         xa_dict[file_name] = xr.load_dataset(file_path, engine="cfgrib")
#     out = xr.merge([array for array in xa_dict.values()], compat="override")
#     # save as new file
#     nc_file_name = ".".join((grib_dir_name, "nc"))
#     save_file_path = "/".join((grib_dir_path, nc_file_name))
#     out.to_netcdf(path=save_file_path)
#     print(f"{nc_file_name} saved successfully")


def determine_ecmwf_values_from_points_df(
    xa_dict: dict,
    weather_keys: list[str],
    df_points: pd.DataFrame,
    distance_buffer: float = 2
) -> pd.DataFrame:
    """Determine the maximum weather values from xarray datasets for each point in a pd.DataFrame

    Parameters
    ----------
    xa_dict : dict
        A dictionary containing xarray datasets as values, disaster names as keys
    weather_keys : list[str]
        A list of strings representing weather keys to determine maximum values
    df_points : pd.DataFrame
        A pandas DataFrame containing latitude, longitude, and disaster name columns
    distance_buffer : float
        A float value representing the distance buffer in degrees

    Returns
    -------
    pd.DataFrame
        pd.DataFrame containing the maximum weather values for each point in df_points
    """
    dictionary_list = []

    df_xbd_points_grouped = df_points.groupby("disaster_name")
    for event_name, group in df_xbd_points_grouped:

        # drop first and last dates (avoids nans from api calling)
        first_date, last_date = xa_dict[event_name].time.values.min(), xa_dict[event_name].time.values.max()
        event_xa = xa_dict[event_name].drop_sel(time=[first_date, last_date])
        df = event_xa.to_dataframe()
        # df = xa_dict[event_name].to_dataframe()

        # remove missing values (i.e. lat-lons in ocean)
        df_nonans = df.dropna(how="any")
        # assign latitude longitude multiindices to columns for easier access
        df_flat = df_nonans.reset_index()

        # coarse df spatial limitation for faster standardisation
        av_lon, av_lat = group.lon.mean(), group.lat.mean()
        df_flat = general_df_utils.limit_df_spatial_range(df_flat, [av_lat, av_lon], distance_buffer=distance_buffer)
        df_flat = general_df_utils.standardise_df(df_flat)

        # iterate through each row in the xbd event df
        for i, row in tqdm(group.iterrows(), total=len(group)):
            poi = Point(row.lon, row.lat)
            # further restrict df for specific point
            df_flat_specific = general_df_utils.limit_df_spatial_range(df_flat, [row.lat, row.lon], min_number=1)
            closest_ind = general_df_utils.find_index_closest_point_in_col(poi, df_flat_specific, "geometry")
            # need to find lon, lat of closest point rather than the index
            # TODO: not sure that max is most relevant for all – parameterisation
            df_maxes = df_flat_specific[df_flat_specific["geometry"] == df_flat_specific.loc[closest_ind]["geometry"]]
            maxs = df_maxes[weather_keys].abs().max()

            # generate dictionary of weather values
            dict_data = {k: maxs[k] for k in weather_keys}
            dict_data["xbd_index"] = row.name
            dict_data["name"] = event_name

            dictionary_list.append(dict_data)

    return pd.DataFrame.from_dict(dictionary_list)


def generate_api_dict(
    weather_params: list[str],
    time_info_dict: dict,
    area: list[float],
    format: str
) -> dict:
    """Generate api dictionary format for single month of event"""

    api_call_dict = {
        "variable": weather_params,
        "area": area,
        "format": format
    } | time_info_dict

    return api_call_dict


def return_full_weather_param_strings(
    dict_keys: list[str]
):
    """Look up weather parameters in a dictionary so they can be entered as short strings rather than typed out in full.
    Key:value pairs ordered in expected importance

    Parameters
    ----------
    dict_keys : list[str]
        list of shorthand keys for longhand weather parameters. See accompanying documentation on GitHub
    """

    weather_dict = {
        "d2m": "2m_dewpoint_temperature", "t2m": "2m_temperature", "skt": "skin_temperature",
        "tp": "total_precipitation",
        "sp": "surface_pressure",
        "src": "skin_reservoir_content", "swvl1": "volumetric_soil_water_layer_1",
        "swvl2": "volumetric_soil_water_layer_2", "swvl3": "volumetric_soil_water_layer_3",
        "swvl4": "volumetric_soil_water_layer_4",
        "slhf": "surface_latent_heat_flux", "sshf": "surface_sensible_heat_flux",
        "ssr": "surface_net_solar_radiation", "str": "surface_net_thermal_radiation",
        "ssrd": "surface_solar_radiation_downwards", "strd": "surface_thermal_radiation_downwards",
        "e": "total_evaporation", "pev": "potential_evaporation",
        "ro": "runoff", "ssro": "sub-surface_runoff", "sro": "surface_runoff",
        "u10": "10m_u_component_of_wind", "v10": "10m_v_component_of_wind",
    }

    weather_params = []
    for key in dict_keys:
        weather_params.append(weather_dict.get(key))

    return weather_params


def generate_times_from_start_end(
    start_end_dates: list[tuple[pd.Timestamp]]
) -> dict:
    """Generate dictionary containing ecmwf time values from list of start and end dates.

    TODO: update so can span multiple months accurately (will involve several api calls)
    """

    # padding dates of interest + 1 day on either side to deal with later nans
    dates = pd.date_range(start_end_dates[0]-pd.Timedelta(1, "d"), start_end_dates[1]+pd.Timedelta(1, "d"))
    years, months, days, hours = set(), set(), set(), []
    # extract years from time
    for date in dates:
        years.add(str(date.year))
        months.add(pad_number_with_zeros(date.month))
        days.add(pad_number_with_zeros(date.day))

    for i in range(24):
        hours.append(f"{i:02d}:00")

    years, months, days = list(years), list(months), list(days)

    time_info = {"year": years, "month": months[0], "day": days, "time": hours}

    return time_info


def fetch_era5_data(
    weather_params: list[str],
    start_end_dates: list[tuple[pd.Timestamp]],
    areas: list[tuple[float]],
    download_dest_dir: str,
    format: str = "grib"
) -> None:
    """Generate API call, download files, merge xarrays, save as new pkl file.

    Parameters
    ----------
    weather_params : list[str]
        list of weather parameter short names to be included in the call
    start_end_dates : list[tuple[pd.Timestamp]]
        list of start and end date/times for each event
    areas : list[tuple[float]]
        list of max/min lat/lon values in format [north, west, south, east]
    download_dest_dir : str
        path to download destination
    format : str = 'grib'
        format of data file to be downloaded

    """
    # initialise client
    c = cdsapi.Client()

    for i, dates in enumerate(start_end_dates):
        # create new folder for downloads
        dir_name = "_".join((
            dates[0].strftime("%d-%m-%Y"), dates[1].strftime("%d-%m-%Y")
            ))
        dir_path = guarantee_existence(os.path.join(download_dest_dir, dir_name))

        time_info_dict = generate_times_from_start_end(dates)

        for param in weather_params:
            # generate api call info TODO: put into function
            api_call_dict = generate_api_dict(param, time_info_dict, areas[i], format)
            file_name = f"{param}.{format}"
            dest = "/".join((dir_path, file_name))
            # make api call
            try:
                c.retrieve(
                    "reanalysis-era5-land",
                    api_call_dict,
                    dest
                )
            # if error in fetching, limit the parameter
            except TypeError():
                print(f"{param} not found in {dates}. Skipping fetching, moving on.")

        # load in all files in folder
        file_paths = "/".join((dir_path, f"*.{format}"))

        xa_dict = {}
        for file_path in tqdm(glob.glob(file_paths)):
            # get name of file
            file_name = file_path.split("/")[-1]
            # read into xarray
            xa_dict[file_name] = xr.load_dataset(file_path, engine="cfgrib")

        # merge TODO: apparently conflicting values of "step". Unsure why.
        out = xr.merge([array for array in xa_dict.values()], compat="override")
        # save as new file
        nc_file_name = ".".join((dir_name, "nc"))
        save_file_path = "/".join((download_dest_dir, nc_file_name))
        out.to_netcdf(path=save_file_path)
        print(f"{nc_file_name} saved successfully")


def geoddist(
    p1: list[float],
    p2: list[float]
):
    """Determines the distance between points p1 and p2

    Parameters
    ----------
    p1 : list(float)
        format [lat1, lon1]
    p2 : list(float)
        format [lat2, lon2]
    """
    return geopy.distance.Geodesic.WGS84.Inverse(p1[1], p1[0], p2[1], p2[0])["s12"]


def generate_xbd_event_xa_dict(
    nc_dir_path: str,
    df_xbd_hurricanes_noaa: pd.DataFrame
) -> dict:
    """Generate a dictionary of xarray datasets for each hurricane event, where each dataset
    contains the relevant variables (latitude, longitude, wind speed, etc.) from the
    corresponding netCDF file in the specified directory.

    Parameters
    ----------
    nc_dir_path : str
        Path to the directory containing the netCDF files
    df_xbd_hurricanes_noaa : pd.DataFrame
        DataFrame containing information about the relevant hurricanes, including their names and dates

    Returns
    -------
    dict
        A dictionary where the keys are the names of the hurricane events and the values are the corresponding
        xarray datasets
    """

    xa_dict = {}
    event_info = {
        "FLORENCE": "14-09-2018",
        "HARVEY": "26-08-2017",
        "MATTHEW": "04-10-2016",
        "MICHAEL": "10-10-2018"
    }
    # load in all .nc files in folder
    file_names = [fl for fl in os.listdir(nc_dir_path) if fl.endswith(".nc")]

    # assign xarrays to labelled dictionary
    for event_name, date in tqdm(event_info.items()):
        # find index of file with date matching event
        index = [idx for idx, s in enumerate(file_names) if date in s][0]
        # generate file path from file name
        file_path = "/".join((nc_dir_path, file_names[index]))
        # assign to dictionary key with correct event name
        xa_dict[event_name] = xr.load_dataset(file_path)

    return xa_dict


def return_relevant_event_info(
    df_point_obs: pd.DataFrame,
    df_xbd_hurricanes_noaa: pd.DataFrame,
    distance_buffer: float = 5,
    verbose: bool = True
) -> tuple[dict[Any, list[list[Any]]], list[list[Any]], list[list[Any]]]:
    """Return the date and geography spans relevvant to each hurricane event in a format conducive to ECMWF API call

    Parameters
    ----------
    df_point_obs : pd.DataFrame
        pd.DataFrame of xbd observations
    df_xbd_hurricanes_noaa : pd.DataFrame
        pd.DataFrame of NOAA HURDAT2 Best Track observations for xbd hurricanes
    distance_buffer : float = 2.5
        number of degrees from mean of observed lat/lons to count as important to the damage. N.B. this is fairly
        arbitrary, and may require expansion when parameterisation of weather is considered
    verbose : bool = True
        if verbose is True, also prints the result

    Returns
    -------
    dict
        dictionary of format {"EVENT_NAME", [[start_time, end_time], [north, west, south, east]]} for each event
    """

    info_dict = {}

    for event_name in df_xbd_hurricanes_noaa.name.unique():
        mean_obs_lat = df_point_obs[df_point_obs["disaster_name"] == event_name]["lat"].mean()
        mean_obs_lon = df_point_obs[df_point_obs["disaster_name"] == event_name]["lon"].mean()

        restricted_event_df = general_df_utils.limit_df_spatial_range(
            df_xbd_hurricanes_noaa[df_xbd_hurricanes_noaa.name == event_name],
            [mean_obs_lat, mean_obs_lon],
            distance_buffer)

        dates = [restricted_event_df.date.min(), restricted_event_df.date.max()]

        # coordinates to maximise
        north, east = (mean_obs_lat+distance_buffer), (mean_obs_lon+distance_buffer)
        # coordinates to minimise
        south, west = (mean_obs_lat-distance_buffer), (mean_obs_lon-distance_buffer)

        maximised, minimised = maximise_area_through_rounding([north, east], [west, south])
        event_area = [maximised[0], minimised[0], minimised[1], maximised[1]]

        info_dict[event_name] = [dates, event_area]
        if verbose:
            print(f"event_name: {event_name}")
            print(f"min_date: {dates[0]}, max_date: {dates[1]}, event_area: {event_area}")

    # reformat for api call
    start_end_dates, areas = [], []
    for k in info_dict:
        start_end_dates.append(info_dict[k][0])
        areas.append(info_dict[k][1])

    return info_dict, start_end_dates, areas


def maximise_area_through_rounding(
    maximise: list[float],
    minimise: list[float]
) -> tuple[list[Any], list[Any]]:
    """Generate an area as large as possible by rounding up/down dependent on sign of coordinate

    Parameters
    ----------
    maximise : list[float]
        e.g. [north, east]
    minimise : list[float]
        e.g. [south, west]

    Returns
    -------
    tuple[list]:
        of maximised/minimised values. For above example this would be: ([north, east], [south, west])
    """

    maximised = []
    minimised = []
    for coord in maximise:
        if coord <= 0:
            max_coord = np.floor(coord)
        else:
            max_coord = np.ceil(coord)
        maximised.append(max_coord)

    for coord in minimise:
        if coord <= 0:
            min_coord = np.ceil(coord)
        else:
            min_coord = np.floor(coord)
        minimised.append(min_coord)

    return maximised, minimised


def save_pkl_to_structured_dir(
    df_to_pkl: pd.DataFrame, 
    pkl_name: str
) -> None:
    """Save pkl file based on name to correct directory
    
    Parameters
    ----------
    df_to_pkl : pd.DataFrame
        pd.DataFrame to be pickled
    pkl_name : str
        name of output pkl file
    
    Returns
    -------
    None
    """

    if pkl_name == "noaa_xbd_hurricanes.pkl" or pkl_name == "noaa_hurricanes.pkl":
        save_dir_path = directories.get_noaa_data_dir()

    elif pkl_name == "ecmwf_params.pkl":
        save_dir_path = directories.get_ecmwf_data_dir()

    elif pkl_name == "df_xbd_points.pkl":
        save_dir_path = directories.get_xbd_dir()

    else:
        raise ValueError(f"Unrecognised pkl name: {pkl_name}")

    save_dest = os.path.join(save_dir_path, pkl_name)
    df_to_pkl.to_pickle(save_dest)
    print(f"{pkl_name} saved successfully to directory: {save_dir_path}")

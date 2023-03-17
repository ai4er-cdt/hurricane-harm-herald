import pandas as pd
import urllib
import numpy as np
from tqdm import tqdm
from h3.dataloading import general_df_utils
import os
import re


def find_fetch_closest_station_files(
    df_xbd_points: pd.DataFrame,
    df_noaa: pd.DataFrame,
    df_stations: pd.DataFrame,
    download_dest_dir: str,
    time_buffer: list[float, str] = [1, 'd'],
    min_number: int = 1,
    distance_buffer: float = None,
) -> pd.DataFrame:
    """Fetch csvs corresponding to closest weather stations from each xBD data point.

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
    df_xbd_points[['event_start', 'event_end']] = np.nan
    df_xbd_points[['closest_stations', 'stations_lat_lons']] = np.nan

    # group by event in df_xbd_points
    df_xbd_points_grouped = df_xbd_points.groupby('disaster_name')
    # for each group in df_xbd_points:
    for name, group in df_xbd_points_grouped:
        # calculate start and end of event
        # df_event_weather = df_noaa[df_noaa['name'] == name]
        start, end = calculate_first_last_dates_from_df(group, time_buffer)
        # limit stations df to those operational within +/- 1 time_buffer either side of event
        df_station_time_lim = df_stations[
            (df_stations['begin'] <= start) & (df_stations['end'] >= end)]

        ignore_csvs = []
        # for each xbd observation in group
        for index, obs in tqdm.tqdm(group.iterrows(), total=len(group)):
            # limit stations spatially
            obs_lat_lons = [obs['lat'], obs['lon']]
            df_station_spatial_time_lim = general_df_utils.limit_df_spatial_range(
                df_station_time_lim, obs_lat_lons, min_number, distance_buffer)

            stations_list = []
            station_no = 0
            while len(stations_list) < min_number:
                # find closest weather station(s) to current weather station (allow closest N, or within limit)
                try:
                    station_index = general_df_utils.find_index_closest_point_in_col(
                        group['geometry'].loc[index], df_station_spatial_time_lim, 'geometry', which_closest=station_no)
                    # get weather station csv filename
                    csv_filename = df_station_spatial_time_lim['csv_filenames'].loc[station_index]
                except: # noqa
                    df_station_spatial_time_lim = general_df_utils.limit_df_spatial_range(
                        df_station_time_lim, obs_lat_lons, len(df_station_spatial_time_lim)+1)
                    station_index = general_df_utils.find_index_closest_point_in_col(
                        group['geometry'].loc[index], df_station_spatial_time_lim, 'geometry', which_closest=station_no)
                    # get weather station csv filename
                    csv_filename = df_station_spatial_time_lim['csv_filenames'].loc[station_index]

                event_year = start.year
                url = generate_station_url(event_year, csv_filename)

                # executes if weather station not already downloaded; if file in ignore, reloop to next-closest station
                if not '/'.join((str(event_year), csv_filename)) in ignore_csvs:
                    # if file doesn't exist, append to ignore and reloop
                    # if file not downloaded
                    if not check_is_file_downloaded(csv_filename, download_dest_dir):
                        try:
                            download_dest = download_dest_dir + '.'.join((csv_filename, 'csv'))
                            urllib.request.urlretrieve(url, download_dest)
                            stations_list.append(csv_filename)
                        except: # noqa
                            ignore_csvs.append('/'.join((str(event_year), csv_filename)))
                    else:
                        stations_list.append(csv_filename)
                station_no += 1

            # append list of stations
            df_xbd_points['closest_stations'].iloc[index] = stations_list
            # append start and end dates
            df_xbd_points['event_start'].iloc[index] = start
            df_xbd_points['event_end'].iloc[index] = end

        # remove station rows which don't exist
        df_stations = df_stations.loc[~df_stations['csv_filenames'].isin(ignore_csvs)]

    return df_xbd_points


def generate_station_url(
    event_year: str,
    csv_filename: str
) -> str:
    """Generate weather station metadata .csv file URL"""
    URL_START = 'https://www.ncei.noaa.gov/data/global-hourly/access/'
    return URL_START + '/'.join((str(event_year), csv_filename)) + '.csv'


def check_is_file_downloaded(
    csv_filename: str,
    download_dest_dir: str
) -> bool:
    """True if already downloaded, False if not"""
    potential_file_path = '/'.join((download_dest_dir, csv_filename)) + '.csv'
    if os.path.exists(potential_file_path):
        # downloaded
        return True
    else:
        return False
        print(f'{csv_filename} already downloaded.')


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
        pd.read_csv(stations_meta_csv_file_path, parse_dates=['BEGIN', 'END']))
    # remove stations with key data missing
    df_stations_valid = df_stations_all.dropna(subset=['lat', 'lon', 'usaf', 'wban'])
    # limit stations to latitude-longitude range of interest
    df_stations_lim = general_df_utils.exclude_df_rows_by_range(
        df_stations_valid, ['lat', 'lon'], [lat_lon_range[0], lat_lon_range[1]])
    # generating filename of hourly weather data
    df_stations = general_df_utils.concat_df_cols(df_stations_lim, 'csv_filenames', ['usaf', 'wban'])

    return df_stations


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
    with open(noaa_meta_txt_file_path, 'r') as noaa_txt_file:
        noaa_data = noaa_txt_file.read().strip().split('\n')

    reformatted_noaa_data = preprocess_noaa_textfile(noaa_data)

    noaa_longform_col_names = [
        'tag', 'name', 'num_entries', 'date', 'time', 'record_id', 'sys_status',  'lat', 'long', 'max_sust_wind',
        'min_p', 'r_ne_34', 'r_se_34', 'r_nw_34', 'r_sw_34', 'r_ne_50', 'r_se_50', 'r_nw_50', 'r_sw_50',
        'r_ne_64', 'r_se_64', 'r_nw_64', 'r_sw_64', 'r_max_wind'
    ]

    noaa_df = reformat_noaa_df(
        pd.DataFrame([el.split(',') for el in reformatted_noaa_data], columns=noaa_longform_col_names))

    if xbd_hurricanes_only:
        # Restrict NOAA data to xbd events
        xbd_hurricane_names = ['MICHAEL', 'MATTHEW', 'FLORENCE', 'HARVEY']
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


# convLatLong
def convert_lat_lon(
    coord: str
) -> str:
    """Convert lat/long of type 00N/S to +/-"""

    if 'S' in coord or 'W' in coord:
        val = '-' + (coord.translate({ord(i): '' for i in 'SW'})).strip()
        return val
    else:
        return coord.translate({ord(i): '' for i in 'NE'})


# checkThreshold
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


def calculate_first_last_dates_from_df(
    df: pd.DataFrame,
    time_buffer: tuple[float, str] = [0, 'h'],
    date_col_name: list[str] = None
) -> tuple[pd.Timestamp]:
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
                    df[col_name] = df[col_name].astype('datetime64[ns]')
            # index = find_first_timestamp_index(col_types)
            # date_col = df.columns.tolist()[index]
            date_col = df.columns[df.apply(
                pd.api.types.is_datetime64_any_dtype)].tolist()[0]
        except TypeError():
            print('No column containing datetime64 objects found')
    else:
        # check if column provided contains datetime objects
        if pd.api.types.is_datetime64_any_dtype(df[date_col_name]):
            date_col = df[date_col_name]
        else:
            raise ValueError(
                'Column provided as date_col_name does not contain datetime objects')

    # if date column type has a timezone detailed
    if pd.api.types.is_datetime64tz_dtype(df[date_col]):
        # convert the 'dates' column to naive timestamps
        df[date_col] = df[date_col].dt.tz_localize(None)

    # generate time buffer
    delta = pd.Timedelta(time_buffer[0], time_buffer[1])

    # find minimum and maximum date values
    start = df[date_col].min() - delta
    end = df[date_col].max() + delta

    return start, end


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

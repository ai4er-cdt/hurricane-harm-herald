from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from typing import Any
import re
import numpy as np
import shapely
import os
import glob
import geopy
import cartopy as cart
import xarray as xr
import tqdm
import cdsapi


# standardiseDfs
def standardise_df(
    df: pd.DataFrame,
    date_cols: list[str] = None,
    new_point_col_name: str = 'geometry'
) -> pd.DataFrame:
    """Apply various formatting functions to make any df behave as you'd expect.

    Parameters
    ----------
    df : pd.DataFrame
        any pandas df
    date_cols : list[str], optional
        list of column names containing date values. Default is None.

    Returns
    -------
    pd.DataFrame
        reformatted pd.DataFrame object
    """
    # makeall headers lower case
    df.columns = df.columns.str.lower()
    # remove any whitespace from headers
    df.columns = df.columns.str.replace(' ', '_')
    # if any columns with dates provided
    if date_cols:
        df[date_cols] = df[date_cols].apply(pd.to_datetime)

    # rename generic 'name' column to 'disaster_name'
    df.rename(columns={'name': 'disaster_name'}, inplace=True)

    if 'geometry' in df.columns:
        # if geometry column not containing shapely Point objects
        if not type(df.geometry.iloc[0]) == shapely.geometry.point.Point:
            df.geometry = df.geometry.apply(lambda x: convert_point_string_to_point(x))
        # generate lat-lon columns from any Point objects
        df = generate_lat_lon_from_points_cols(df, ['geometry'])

    # TODO: can I tidy this up?
    if set(['latitude', 'longitude']).issubset(df.columns):
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

    if set(['lat', 'lon']).issubset(df.columns):
        # make geometry column of shapely Point objects
        df = points_from_df_lat_lon_cols(df, new_point_col_name)
    # common variation
    elif set(['lat', 'long']).issubset(df.columns):
        df.rename(columns={'long': 'lon'}, inplace=True)
        df = points_from_df_lat_lon_cols(df, new_point_col_name)

    return df


# pointsFromLatLon
def points_from_df_lat_lon_cols(
    df: pd.DataFrame,
    point_col_name: str = 'geometry'
) -> pd.DataFrame:
    """Generate shapely.geometry.point.Point object for each row of a df and assign to a new column headed
    'point_col_name'. N.B. df must contain columns titled 'lat' and 'lon'
    """
    df[point_col_name] = df.apply(lambda row: shapely.geometry.point.Point(
        row['lon'], row['lat']), axis=1)

    return df


# checklistLengthsEqual
def check_lists_equal_length(
    lists: list[list[Any]]
):
    """Checks whether all lists in 'lists' are the same length: raises error if
    not.

    Parameters
    ----------
    lists : list[ list[Any] ]
        list of lists containing any type of data
    """

    # check lists same lengths
    it = iter(lists)
    the_len = len(next(it))
    if not all(len(el) == the_len for el in it):
        raise ValueError('Not all lists have same the length')


# symmetricalExclude
def exclude_df_rows_symmetrically_around_value(
    df: pd.DataFrame,
    col_names: list[str],
    poi: list[float] | list[pd.Timestamp],
    buffer_val: list[float] | list[tuple[float, str]]
) -> pd.DataFrame:
    """Return a pd.DataFrame which excludes rows outside a range of +/- 1 buffer.
    Buffer can be floats objects, or can specify a period of time. Handy e.g.
    for excluding stations for which there is no weather data within the period
    of interest.

    Parameters
    ----------
    df : pd.DataFrame
        pd.DataFrame containing values to potentially exclude
    col_names : list[str]
        list of strings specifying the names of the columns of interest
    poi : [ list[float], list[pd.Timestamp] ]
        points of interest (value about which any exclusion will be centred).
        One value for each relevant column.
    buffer : [ list[float], list[tuple[float,str]] ]
        distance from poi to be excluded. In the case that poi is a Timestamp
        object, a string specifying the unit of time is necessary e.g. 'h' for
        hours (either as a tuple or list). See
        https://pandas.pydata.org/docs/reference/api/pandas.Timedelta.html for
        more info. One value for each relevant column.

    Returns
    -------
    pd.DataFrame excluding values outside of provided ranges
    """

    # check lists same lengths
    check_lists_equal_length([col_names, poi, buffer_val])

    for i, col in enumerate(col_names):

        if type(poi[i]) == pd.Timestamp:
            # specify the buffer as a Timestamp object (separating time and unit)
            buffer = pd.Timedelta(buffer_val[i][0], buffer_val[i][1])
        else:
            buffer = buffer_val[i]
        # restrict to only observations within the range
        df = df[df[col].between(poi[i]-buffer, poi[i]+buffer)]

    return df


# rangeExclude
def exclude_df_rows_by_range(
    df: pd.DataFrame,
    col_names: list[str],
    value_bounds: list[tuple[float]] | list[list[float]],
    buffer: list[float] | list[tuple[float, str]] = 0
) -> pd.DataFrame:
    """Return pd.DataFrame composed of only rows containing only values in
    columns listed in col_names within the range of value_bounds +/- optional
    buffer amount. Handy for restricting large dataframes based on date ranges
    (must specify bounds as pd.Timestamp objects), or lat/lon ranges.

    Parameters
    ----------
    df : pd.DataFrame
        df to limit
    col_names : list[str]
        e.g. ['col1', ..., 'colN']
        list of column names to be restricted by their relevant...
    value_bounds : list[[tuple[float] | list[list[float]]
        e.g. [ (start_val1,end_val1), ..., (start_valN,end_valN) ]
        list of tuples (or lists) specifying minimum and maximum values to allow
    buffer : list[float] | list[tuple[float,str]] = 0
        add buffer on either side of value_bounds. Defaults to no buffer. Useful
        for specifying weather station observations must exist some time before
        and after the event of interest

    Returns
    -------
    restricted pd.DataFrame object (sub-set of original df)
    """
    # check lists same lengths
    check_lists_equal_length([col_names, value_bounds])

    for i, col in enumerate(col_names):

        if type(value_bounds[i][0]) == pd.Timestamp:
            # specify the buffer as a Timestamp object (separating time and unit)
            buffer = pd.Timedelta(buffer[0], buffer[1])

        df = df[df[col].between(
            min(value_bounds[i])-buffer, max(value_bounds[i])+buffer)]

    return df


# concatDfsCols
def concat_df_cols(
    df: pd.DataFrame,
    concatted_col_name: str,
    cols_to_concat: list[str],
    delimiter: str = ''
) -> pd.DataFrame:
    """Concatenate columns in a pd.DataFrame into a new column of strings linked
    by 'delimiter'.capitalize()

    Parameters
    ----------
    df : pd.DataFrame
        df containing columns to concatenate
    concatted_col_name : str
        name of new concatenated column
    cols_to_concat : list[str]
        names of columns to concatenate (in desired order)
    delimiter : str = ''
        character to insert in between column values. Defaults to empty string

    Returns
    -------
    pd.DataFrame
        with additional concatted column
    """
    df[concatted_col_name] = df[cols_to_concat].astype(str).apply(
        delimiter.join, axis=1)

    return df


###
# NOAA 6-HOURLY DATASET PROCESSING FUNCTIONS
# these are rather specific to the task of reading in
# 'hurdat2-1851-2021-100522.txt' but some could be generalised if necessary.
###


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


# dfPreprocess
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


# reformatDf
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


# pathExists
def get_path(
    path: str
):
    """
    Check that path to file exists. May have different functionality in colab
    """
    if os.path.exists(path):
        return path
    else:
        raise FileNotFoundError(f'{path} does not exist')


# pointStrToPoint
def convert_point_string_to_point(
    point_string: str
) -> shapely.geometry.Point:
    """Convert string of Point object to actual Point object

    Parameters
    ----------
    point_string : str

    Returns
    -------
    shapely.geometry.Point
    """
    coords = [float(coord) for coord in re.findall(r'-?\d+\.\d+', point_string)]

    return shapely.geometry.Point(coords)


# latLonFromPoints
def generate_lat_lon_from_points_cols(
    df: pd.DataFrame,
    points_cols: list[str]
) -> pd.DataFrame:
    """Generate a column(s) of lat and lon from column(s) of shapely.Point
    objects

    Parameters
    ----------
    df: pd.DataFrame
        pd.DataFrame containing column(s) of shapely.Point objects
    points_cols: list[str]
        names of columns to convert to lat/lon. Chosen not to find the columns as
        default (by dtype) since might only want to convert one.

    Returns
    -------
    pd.DataFrame
        containing new lat and lon columns
    """

    for i, col in enumerate(points_cols):
        if len(points_cols) == 1:
            lon_col_name = 'lon'
            lat_col_name = 'lat'
        else:
            lon_col_name = f'lon{i+1}'
            lat_col_name = f'lat{i+1}'

        df[lon_col_name] = df[col].apply(lambda p: p.x)
        df[lat_col_name] = df[col].apply(lambda p: p.y)

    return df


# calcDistance
def calc_distance_between_df_cols(
    df: pd.DataFrame,
    cols_compare: list[tuple[str]] | list[list[str]],
    new_col_name: str = 'distance'
) -> pd.DataFrame:
    """Calculate the geodesic distance between sets of lat/lon values. See
    https://geopy.readthedocs.io/en/stable/#module-geopy.distance for more info.

    Parameters
    ----------
    df: pd.DataFrame
        df containing two pairs of lat/lon values
    cols_compare: list[[tuple[str]] | list[list[str]]
        list of columns of lat/lon values. Inputted as pairs as a tuple or list

    Returns
    -------
    pd.DataFrame
        copy of df with an extra 'distance' column
    """
    if not len(cols_compare) == 2:
        raise ValueError(
            '''Cannot compare more or fewer than two sets of lat/lon values at a time''')

    df[new_col_name] = df.apply(
        lambda x: geopy.distance.geodesic(
            (x[cols_compare[0][0]], x[cols_compare[0][1]]),
            (x[cols_compare[1][0]], x[cols_compare[1][1]])).km, axis=1)

    return df


# closestPointsIndices
def find_index_closest_point_in_col(
    poi: shapely.Point,
    points_df: pd.DataFrame,
    points_df_geom_col: str,
    which_closest: int = 0
) -> int:
    """Find the df index of the closest point object to poi in the df object.

    Parameters
    ----------
    poi : shapely.Point
        point of interest (shapely.Point object)
    points_df : pd.DataFrame
        dataframe containing a column of shapely.Point objects
    points_df_geom_col : str
        name of column of shapely.Point objects
    which_closest : int = 1
        if 1 (default), find closest. For any other N, find the Nth closest

    Returns
    -------
    int object relating to index of points_df df
    """
    distances = points_df[points_df_geom_col].apply(lambda x: poi.distance(x))
    s = sorted(set(distances))
    return distances[distances == s[which_closest]].index[0]


# def findClosestPointIndices(
#     df1: pd.DataFrame,
#     df1_grouping_col: str,
#     df2: pd.DataFrame,
#     df2_grouping_col: str
# ) -> pd.DataFrame:
#     """Find the closest shapely.Point object in df2 to each shapely.Point object
#     in df1, and append the index of the relevant row to df1.as_integer_ratio.
#     N.B. distance found in Euclidean space (not the actual distance between
#     lat/lons: but result is the same and achieved quicker.

#     Parameters
#     ----------
#     df1 : pd.DataFrame
#         pd.DataFrame object containing a list of shapely.Point objects
#     df1_grouping_col : str
#         column name of variable to group df1 by e.g. name of disaster
#     df2 : pd.DataFrame
#         pd.DataFrame object containing a list of shapely.Point objects
#     df2_grouping_col : str
#         column name of variable to group df2 by e.g. name of disaster. Almost
#         always the same as df1_grouping_col

#     Returns
#     -------
#     pd.DataFrame
#         copy of df1 with the indices of the rows containing closest points in
#         df2 appended in a shiny new 'index_closest' column
#     """

#     # find geometry column in df1 and df2
#     df1_points_col = df1.columns[df1.dtypes == 'geometry'].tolist()[0]
#     df2_points_col = df2.columns[df2.dtypes == 'geometry'].tolist()[0]

#     df1_grouped = df1.groupby(df1_grouping_col)
#     # pre-assign column of values
#     df1['index_closest'] = np.nan
#     df2 = df2.reset_index()

#     for grouping, group in df1_grouped:
#         # restrict iteration's search based on grouping parameter e.g. by
#         # event name
#         df1_restricted = df1.query('{} == "{}"'.format(
#             df1_grouping_col, grouping))
#         df2_restricted = df2.query('{} == "{}"'.format(
#             df2_grouping_col, grouping))

#         for index, row in tqdm.tqdm(group.iterrows()):
#             # for each row in df1, find closest shapely.Point in df2 and append
#             df1['index_closest'].iloc[index] = closestPointsIndices(
#                 df1_restricted[df1_points_col].loc[index],
#                 df2_restricted, df2_points_col)

#     # make the column values integers for prettiness (doesn't affect
#     # functionality)
#     df1 = df1.astype({'index_closest': int})

#     return df1


# mostRecentEventsByName
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


# calcStartEndTimes
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


# calcDfColMeanValues
def calc_means_df_cols(
    df: pd.DataFrame,
    col_names: list[str]
) -> pd.DataFrame:
    """Return mean values of prescribed columns in df

    Parameters
    ----------
        df : pd.DataFrame
        col_names : list[str]
            list of columns to calculate mean
    Returns
    -------
        list[float] of mean value of each column
    """
    means = []
    for col in col_names:
        means.append(df[col].mean())
    return means


# limitDfSpatially
def limit_df_spatial_range(
    df: pd.DataFrame,
    centre_coords: list[float] | tuple[float],
    min_number: int = None,
    distance_buffer: float = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Restrict df to within +/- a lat-lon distance, or to a min_number number
    of rows.

    Parameters
    ----------
    df : pd.DataFrame
        df containing 'lat' and 'lon' columns
    centre_coords : list[float] | tuple[float]
        geographical centre about which to restrict df (['lat', 'lon'])
    min_number : int = None
        minimum number of rows in df to be returned
    distance_buffer : float = None
        distance from geographical centre within which points in df should be
        returned
    verbose : bool = False (don't show message re expansion of distance_buffer)
        choose whether or not to show that distance_buffer was expanded

    Returns
    -------
    pd.DataFrame
        spatially limited df
    """
    if not set(['lat', 'lon']).issubset(df.columns) | set(['latitude', 'longitude']).issubset(df.columns):
        raise ValueError('Columns by name of lat and lon not found in df')

    # slight hack to avoid standardising huuge dfs in emwcf
    if set(['latitude', 'longitude']).issubset(df.columns):
        df.rename(columns={'latitude': 'lat', 'longitude': 'lon'}, inplace=True)

    # choosing to find closest N points at any distance away
    if distance_buffer is None:
        # set arbitrarily small distance buffer
        distance_buffer = 0.1
        df_spatial_lim = exclude_df_rows_symmetrically_around_value(
            df, ['lat', 'lon'], centre_coords,
            [distance_buffer, distance_buffer])
        # expand distance buffer until minimum number reached
        while len(df_spatial_lim) <= min_number:
            distance_buffer += 0.1
            df_spatial_lim = exclude_df_rows_symmetrically_around_value(
                df, ['lat', 'lon'], centre_coords,
                [distance_buffer, distance_buffer])
            if verbose is True:
                print(f'Spatial search expanded to +/- {distance_buffer} degrees')
    # if choosing to find closest stations only up to a certain distance
    else:
        df_spatial_lim = exclude_df_rows_symmetrically_around_value(
            df, ['lat', 'lon'], centre_coords,
            [distance_buffer, distance_buffer])

    return df_spatial_lim


# checkExistswriteToPickle
def write_df_to_pkl(
    target_dir: str,
    filename: str,
    df: pd.DataFrame
):
    """Check if file exists at write_location before writing to pkl file

    Parameters
    ----------
    target_dir : str
        directory to which pkl file should be written
    filename : str
        name of pkl file (without extension)
    df : pd.DataFrame
        df to write to pkl
    """

    pkl_filename = '.'.join((filename, 'pkl'))
    write_location = '/'.join((target_dir, pkl_filename))
    # check that file doesn't already exist in target location
    if not os.path.exists(write_location):
        df.to_pickle(write_location)
        print(f'{pkl_filename} written to {target_dir}')
    else:
        print(f'File path already exists. No new file written to {write_location}.')


def generate_cmorph_urls(
    time_buffer_before: pd.Timedelta,
    time_buffer_after: pd.Timedelta,
    df_xbd_points: pd.DataFrame,
    df_xbd_hurricanes_noaa: pd.DataFrame
) -> list[str]:
    """
    Returns a list of URLs specifying .nc weather files according to this
    file structure:
    https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/hourly/0.25deg/

    Parameters
    ----------
    time_buffer_before : pd.Timedelta
        specify the length of time before the central event time to get files
        e.g. 12 days: pd.Timedelta(12,'d')
    time_buffer_after: pd.Timedelta
        specify the length of time after the central event time to get files
        e.g. 2 hours: pd.Timedelta(2,'h')

    Returns
    -------
    list[str]
        each corresponding to a downloadable url
    """

    url_root = 'https://www.ncei.noaa.gov/data/cmorph-high-resolution-global-precipitation-estimates/access/hourly/0.25deg/' # noqa
    file_root = 'CMORPH_V1.0_ADJ_0.25deg-HLY_'
    urls = []

    # find closest geographical point (use average of xbd points)
    df_xbd_points_grouped = df_xbd_points.groupby('disaster_name')
    for name, group in df_xbd_points_grouped:
        # restrict weather df to relevant hurricane
        df_event_weather = df_xbd_hurricanes_noaa[df_xbd_hurricanes_noaa['name'] == name]
        av_lat, av_lon = calc_means_df_cols(group, ['lat', 'lon'])
        # index of weather df for closest hurricane observation to average xbd
        # point location
        noaa_index = find_index_closest_point_in_col(
            shapely.Point(av_lat, av_lon), df_event_weather, 'geometry')

        # get time of point
        mid_event_time = df_event_weather['date'].loc[noaa_index]
        # get time before and after (buffer)
        start = mid_event_time - time_buffer_before
        end = mid_event_time + time_buffer_after
        dates = pd.date_range(start, end, freq='d')

        # generate urls
        for date in dates:
            url_date = '/'.join(
                [str(date.year), pad_number_with_zeros(str(date.month)), pad_number_with_zeros(str(date.day)), ''])
            file_name_no_hour = ''.join(
                [str(date.year), pad_number_with_zeros(str(date.month)), pad_number_with_zeros(str(date.day))])
            # currently getting all hours of the day from start to finish (rather
            # than starting and finishing at a specified hour). This is no problem,
            # since coarse buffer.
            # numbering hours
            for hour in range(24):
                file_name = ''.join((file_name_no_hour, pad_number_with_zeros(str(hour)))) + '.nc'
                # N.B. appending to list via tuples much faster than .append()
                urls += ''.join([url_root, url_date, file_root, file_name]),
    return urls


# Pad_With_Zeros
def pad_number_with_zeros(
    number: str | int
) -> str:
    """
    Add a leading zero to any number, X, into 0X. Useful for generating dates in URL strings.
    """

    if not type(number) == str:
        try:
            number = str(number)
        except ValueError:
            print(f'Failed to convert {number} to string')
    if len(number) == 1:
        number = ''.join(('0', number))

    return number


# station_availability
def station_availability(
    df_stations: pd.DataFrame,
    df_noaa_weather_event: pd.DataFrame,
    time_buffer: list[float, str] = [0, 'h'],
    available: bool = True
) -> pd.DataFrame:
    """
    Filter dataframe by time to return only stations with observation present.
    Defaults to available
    """

    start, end = calculate_first_last_dates_from_df(df_noaa_weather_event, time_buffer)

    if available:
        # return available stations
        df_station_time_lim = df_stations[
            ((df_stations['begin'] <= start) & (df_stations['end'] >= end))]
    else:
        # limit stations df to those operational within +/- 1 time_buffer
        # either side of event
        df_station_time_lim = df_stations[
            ~((df_stations['begin'] <= start) & (df_stations['end'] >= end))]

    return df_station_time_lim


def maximise_area_through_rounding(
    maximise: list[float],
    minimise: list[float]
) -> tuple[list]:
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


def return_relevant_event_info(
    df_point_obs: pd.DataFrame,
    df_xbd_hurricanes_noaa: pd.DataFrame,
    distance_buffer: float = 5,
    verbose: bool = True
) -> dict:
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
        dictionary of format {'EVENT_NAME', [[start_time, end_time], [north, west, south, east]]} for each event
    """

    info_dict = {}

    for event_name in df_xbd_hurricanes_noaa.name.unique():
        mean_obs_lat = df_point_obs[df_point_obs['disaster_name'] == event_name]['lat'].mean()
        mean_obs_lon = df_point_obs[df_point_obs['disaster_name'] == event_name]['lon'].mean()

        restricted_event_df = limit_df_spatial_range(
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
            print(f'event_name: {event_name}')
            print(f'min_date: {dates[0]}, max_date: {dates[1]}, event_area: {event_area}')

    # reformat for api call
    start_end_dates, areas = [], []
    for k in info_dict:
        start_end_dates.append(info_dict[k][0])
        areas.append(info_dict[k][1])

    return info_dict, start_end_dates, areas


def standardise_xbd_obs_df(
    df: pd.DataFrame
) -> pd.DataFrame:
    """Standardise df of xbd point observations. Renames 'disaster_name' column to 'name', and makes each disaster name
    standalone and capitalised e.g. 'FLORENCE' rather than 'hurricane-florence'
    """

    # match naming convention with NOAA for ease of comparison later
    df['disaster_name'] = df['disaster_name'].apply(
        lambda x: x.split('-')[-1]).str.upper()

    return standardise_df(df, date_cols=['capture_date'])


def yes_or_no(question):
    while "Answer is invalid":
        reply = str(input(question + ' (y/n): ')).strip()
        if reply[:1] == 'y':
            return True
        if reply[:1] == 'n':
            return False


def create_dir_if_absent(
    parent_dir_path: str,
    dir_name: str
):
    """Checks if directory by specified path exists, else creates it. Returns directory path."""
    dir = parent_dir_path + '/' + dir_name
    if not os.path.exists(dir):
        os.mkdir(dir)

    return dir


def plot_hurricane_event_tracks(
    df_noaa_hurricanes: pd.DataFrame,
    df_xbd_points: pd.DataFrame = None,
    df_stations: pd.DataFrame = None,
    min_number_weather: int = None
):
    """
    Plot tracks of hurricane events with additional weather observations on a geographic map.

    Parameters
    ----------
    df_noaa_hurricanes : pd.DataFrame
        DataFrame with NOAA hurricane observation data. Must contain columns 'name', 'date', 'tag', and 'geometry'.
    df_xbd_points : pd.DataFrame, default None
        DataFrame with additional weather observation data. Must contain columns 'disaster_name', 'lat', 'lon',
        and 'geometry'. Default is None.
    df_stations : pd.DataFrame, default None
        DataFrame with weather station data. Must contain columns 'name', 'lat', 'lon', and 'geometry'. Default is None.
    min_number_weather : int, default None
        The minimum number of weather observations to be plotted. Default is None.

    Returns
    -------
    None
    """

    # generate number of plots necessary
    num_plots = len(df_noaa_hurricanes.tag.unique())
    num_rows = int(np.ceil(num_plots / 2))

    # fig, axs = plt.subplots(num_rows, 2, figsize=[12*num_rows,12]);
    fig, axs = plt.subplots(num_rows, 2, figsize=[12*num_rows, 12], dpi=300,
                            subplot_kw={'projection': cart.crs.PlateCarree()})

    axs = axs.ravel()

    for i, t in enumerate(df_noaa_hurricanes.tag.unique()):
        plt.rcParams.update({'axes.titlesize': 'medium'})

        # get first occurence of name
        event_name = df_noaa_hurricanes[df_noaa_hurricanes.tag == t].name.iloc[0]
        start_date = df_noaa_hurricanes[df_noaa_hurricanes.tag == t].date.min()

        # formatting
        plot_title = f'{t}: {event_name}\nstart_date: {start_date}'
        axs[i].set_title(plot_title)
        # axs[i].set_xlabel('longitude'), axs[i].set_ylabel('latitude')
        axs[i].set_aspect('equal', adjustable='datalim')
        # gdf_coastlines.plot(ax=axs[i], color='grey', alpha=1, linewidth=0.5)
        axs[i].add_feature(cart.feature.LAND.with_scale('10m'))
        axs[i].add_feature(cart.feature.OCEAN.with_scale('10m'))

        gdf_noaa_hurricanes = gpd.GeoDataFrame(df_noaa_hurricanes[df_noaa_hurricanes['name'] == event_name])
        gdf_noaa_hurricanes.plot(ax=axs[i], color='blue', markersize=100, alpha=0.3, label='NOAA hurricane observation')

        if df_xbd_points is not None:
            gdf_xbd_points = gpd.GeoDataFrame(df_xbd_points[df_xbd_points['disaster_name'] == event_name])
            gdf_xbd_points.plot(ax=axs[i], color='red', markersize=0.5, alpha=1, label='xbd observation')
            av_lat, av_lon = calc_means_df_cols(gdf_xbd_points, ['lat', 'lon'])

            # if limiting plot to a show at least 'min_number_weather' datapoints
            if min_number_weather:
                # ensure that at least 'min_number_weather' hurricane observation points are in the image
                gdf_noaa_hurricanes = limit_df_spatial_range(
                    gdf_noaa_hurricanes, [av_lat, av_lon], min_number=min_number_weather)

            if df_stations is not None:
                gdf_stations = gpd.GeoDataFrame(
                    df_stations, geometry=gpd.points_from_xy(df_stations.lon, df_stations.lat), crs="EPSG:4326")
                # inactive weather stations
                gdf_stations_inactive = station_availability(gdf_stations, gdf_noaa_hurricanes, available=False)
                gdf_stations_inactive.plot(
                    ax=axs[i], color='darkorange', markersize=10, label='inactive weather station')
                # active weather stations
                gdf_stations_active = station_availability(gdf_stations, gdf_noaa_hurricanes, available=True)
                gdf_stations_active.plot(
                    ax=axs[i], color='lime', markersize=10, alpha=1, label='active weather station')

            # generate list of observation coordinates
            coords = list(gdf_xbd_points['geometry'])+list(gdf_noaa_hurricanes['geometry'])
            lats = [pt.y for pt in coords]
            lons = [pt.x for pt in coords]
            # restrict axes limits to zoom in on available data
            axs[i].set_ylim(min(lats), max(lats))
            axs[i].set_xlim(min(lons), max(lons))
            axs[i].legend(loc='upper left')

        axs[i].set_xlabel('Latitude (°)')
        axs[i].set_xlabel('Longitude (°)')
        # xticks, yticks = axs[i].get_xticks(), axs[i].get_yticks()


def restrict_coords(
    av_lon: float,
    av_lat: float,
    weather_lons: list[float],
    weather_lats: list[float],
    buffer: float
) -> tuple[list]:
    """
    Return a tuple of two lists containing longitude and latitude coordinates from the input
    `weather_lons` and `weather_lats` lists that fall within a given distance (`buffer`) from a
    specified location (`av_lon`, `av_lat`)

    Parameters
    ----------
    av_lon : float
        The longitude of the specified location
    av_lat : float
        The latitude of the specified location
    weather_lons : list[float])
        A list of longitude coordinates
    weather_lats : list[float]
        A list of latitude coordinates
    buffer : float
        The maximum distance from the specified location within which to include the weather coordinates

    Returns
    -------
    tuple[list]
        A tuple containing two lists. First list contains longitude coordinates that fall within the given distance
        from the specified location. Second list contains latitude coordinates that fall within the given distance from
        the specified location
    """

    min_lon, max_lon = av_lon-buffer, av_lon+buffer
    min_lat, max_lat = av_lat-buffer, av_lat+buffer

    rest_weather_lons = [num for num in weather_lons if num >= min_lon and num <= max_lon]
    rest_weather_lats = [num for num in weather_lats if num >= min_lat and num <= max_lat]

    return rest_weather_lons, rest_weather_lats


def geoddist(
    p1: list(float),
    p2: list(float)
):
    """Determines the distance between points p1 and p2

    Parameters
    ----------
    p1 : list(float)
        format [lat1, lon1]
    p2 : list(float)
        format [lat2, lon2]
    """
    return geopy.distance.Geodesic.WGS84.Inverse(p1[1], p1[0], p2[1], p2[0])['s12']


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
        'FLORENCE': '14-09-2018',
        'HARVEY': '26-08-2017',
        'MATTHEW': '04-10-2016',
        'MICHAEL': '10-10-2018'
    }
    # load in all .nc files in folder
    file_names = [fl for fl in os.listdir(nc_dir_path) if fl.endswith(".nc")]

    # assign xarrays to labelled dictionary
    for event_name, date in tqdm.tqdm(event_info.items()):
        # find index of file with date matching event
        index = [idx for idx, s in enumerate(file_names) if date in s][0]
        # generate file path from file name
        file_path = '/'.join((nc_dir_path, file_names[index]))
        # assign to dictionary key with correct event name
        xa_dict[event_name] = xr.load_dataset(file_path)

    return xa_dict


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

    df_xbd_points_grouped = df_points.groupby('disaster_name')
    for event_name, group in df_xbd_points_grouped:

        # drop first and last dates (avoids nans from api calling)
        first_date, last_date = xa_dict[event_name].time.values.min(), xa_dict[event_name].time.values.max()
        event_xa = xa_dict[event_name].drop_sel(time=[first_date, last_date])
        df = event_xa.to_dataframe()
        # df = xa_dict[event_name].to_dataframe()

        # remove missing values (i.e. lat-lons in ocean)
        df_nonans = df.dropna(how='any')
        # assign latitude longitude multiindices to columns for easier access
        df_flat = df_nonans.reset_index()

        # coarse df spatial limitation for faster standardisation
        av_lon, av_lat = group.lon.mean(), group.lat.mean()
        df_flat = limit_df_spatial_range(df_flat, [av_lat, av_lon], distance_buffer)
        df_flat = standardise_df(df_flat)

        # iterate through each row in the xbd event df
        for i, row in tqdm.tqdm(group.iterrows(), total=len(group)):
            poi = shapely.geometry.point.Point(row.lon, row.lat)
            # further restrict df for specific point
            df_flat_specific = limit_df_spatial_range(df_flat, [row.lat, row.lon], min_number=1)
            closest_ind = find_index_closest_point_in_col(poi, df_flat_specific, 'geometry')
            # need to find lon, lat of closest point rather than the index
            # TODO: not sure that max is most relevant for all – parameterisation
            df_maxes = df_flat_specific[df_flat_specific['geometry'] == df_flat_specific.loc[closest_ind]['geometry']]
            maxs = df_maxes[weather_keys].abs().max()

            # generate dictionary of weather values
            dict_data = {k: maxs[k] for k in weather_keys}
            dict_data['xbd_index'] = row.name
            dict_data['name'] = event_name

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
        'd2m': '2m_dewpoint_temperature', 't2m': '2m_temperature', 'skt': 'skin_temperature',
        'tp': 'total_precipitation',
        'sp': 'surface_pressure',
        'src': 'skin_reservoir_content', 'swvl1': 'volumetric_soil_water_layer_1',
        'swvl2': 'volumetric_soil_water_layer_2', 'swvl3': 'volumetric_soil_water_layer_3',
        'swvl4': 'volumetric_soil_water_layer_4',
        'slhf': 'surface_latent_heat_flux', 'sshf': 'surface_sensible_heat_flux',
        'ssr': 'surface_net_solar_radiation', 'str': 'surface_net_thermal_radiation',
        'ssrd': 'surface_solar_radiation_downwards', 'strd': 'surface_thermal_radiation_downwards',
        'e': 'total_evaporation', 'pev': 'potential_evaporation',
        'ro': 'runoff', 'ssro': 'sub-surface_runoff', 'sro': 'surface_runoff',
        'u10': '10m_u_component_of_wind', 'v10': '10m_v_component_of_wind',
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
    dates = pd.date_range(start_end_dates[0]-pd.Timedelta(1, 'd'), start_end_dates[1]+pd.Timedelta(1, 'd'))
    years, months, days, hours = set(), set(), set(), []
    # extract years from time
    for date in dates:
        years.add(str(date.year))
        months.add(pad_number_with_zeros(date.month))
        days.add(pad_number_with_zeros(date.day))

    for i in range(24):
        hours.append(f'{i:02d}:00')

    years, months, days = list(years), list(months), list(days)

    time_info = {"year": years, "month": months[0], "day": days, "time": hours}

    return time_info


def fetch_era5_data(
    weather_params: list[str],
    start_end_dates: list[tuple[pd.Timestamp]],
    areas: list[tuple[float]],
    download_dest_dir: str,
    format: str = 'grib'
):
    """Generate API call, download files, merge xarrays, save as new pkl file.

    Parameters
    ----------
    weather_keys : list[str]
        list of weather parameter short names to be included in the call
    start_end_dates : list[tuple[pd.Timestamp]]
        list of start and end date/times for each event
    area : list[tuple[float]]
        list of max/min lat/lon values in format [north, west, south, east]
    download_dest_dir : str
        path to download destination
    format : str = 'grib'
        format of data file to be downloaded

    Returns
    -------
    None
    """
    # initialise client
    c = cdsapi.Client()

    for i, dates in enumerate(start_end_dates):
        # create new folder for downloads
        destination_path = get_path(download_dest_dir)
        dir_name = '_'.join((
            dates[0].strftime("%d-%m-%Y"), dates[1].strftime("%d-%m-%Y")
            ))
        dir_path = create_dir_if_absent(destination_path, dir_name)

        time_info_dict = generate_times_from_start_end(dates)

        for param in weather_params:
            # generate api call info TODO: put into function
            api_call_dict = generate_api_dict(param, time_info_dict, areas[i], format)
            file_name = f'{param}.{format}'
            dest = '/'.join((dir_path, file_name))
            # make api call
            # TODO: is there a nice way to overwrite files of same name, provided they are
            # different? e.g. different area
            try:
                c.retrieve(
                    'reanalysis-era5-land',
                    api_call_dict,
                    dest
                )
            # if error in fetching, limit the parameter
            except TypeError():
                print(f'{param} not found in {dates}. Skipping fetching, moving on.')

        # load in all files in folder
        file_paths = '/'.join((dir_path, f'*.{format}'))

        xa_dict = {}
        for file_path in tqdm.tqdm(glob.glob(file_paths)):
            # get name of file
            file_name = file_path.split('/')[-1]
            # read into xarray
            xa_dict[file_name] = xr.load_dataset(file_path, engine="cfgrib")

        # merge TODO: apparently conflicting values of 'step'. Unsure why.
        out = xr.merge([array for array in xa_dict.values()], compat='override')
        # save as new file
        nc_file_name = '.'.join((dir_name, 'nc'))
        save_file_path = '/'.join((destination_path, nc_file_name))
        out.to_netcdf(path=save_file_path)
        print(f'{nc_file_name} saved successfully')


def find_NOAA_points(
    df_noaa_xbd_hurricanes: pd.DataFrame,
    df_xbd_points: pd.DataFrame
) -> pd.DataFrame:
    """
    Appends the closest weather data from NOAA 6-hourly data to xbd points.

    Parameters
    ----------
    df_noaa_xbd_hurricanes : pd.DataFrame
        Dataframe of NOAA hurricane data.
    df_xbd_points : pd.DataFrame
        Dataframe of xBD points.

    Returns
    -------
    pd.DataFrame
        pd.DataFrame with the closest weather data from NOAA 6-hourly data to each xBD points.
    """

    noaa_indices = []
    xbd_indices = []
    distances = []
    # group by event in df_xbd_points
    df_xbd_points_grouped = df_xbd_points.groupby('disaster_name')
    # for each group in df_xbd_points:
    for name, group in df_xbd_points_grouped:
        df_event_weather = df_noaa_xbd_hurricanes[df_noaa_xbd_hurricanes['name'] == name]

        for index, obs in tqdm.tqdm(group.iterrows(), total=len(group)):
            # find index of noaa observation datapoint closest to xbd point
            noaa_index = find_index_closest_point_in_col(
                group['geometry'].loc[index], df_event_weather, 'geometry')
            noaa_row = df_noaa_xbd_hurricanes.loc[noaa_index]
            # calculate distance between xbd point and noaa observation
            distance = geopy.distance.geodesic((obs['lat'], obs['lon']), (noaa_row['lat'], noaa_row['lon'])).km

            # append to list as tuple (faster than appending as value)
            noaa_indices += noaa_index,
            xbd_indices += index,
            distances += distance,

    # reindex dataframes to prepare for merge
    reindexed_noaa_xbd_hurricanes = df_noaa_xbd_hurricanes.reindex(noaa_indices)
    reindexed_noaa_xbd_hurricanes = reindexed_noaa_xbd_hurricanes.reset_index().rename(
        columns={'index': 'noaa_index'})

    reindexed_xbd_points = df_xbd_points.reindex(xbd_indices)
    reindexed_xbd_points = reindexed_xbd_points.reset_index().rename(columns={'index': 'xbd_index'})

    # rename columns before merge to avoid duplicate column names
    reindexed_noaa_xbd_hurricanes.rename(
        columns={
            'geometry': 'noaa_obs_geometry', 'lon': 'noaa_obs_lon', 'lat': 'noaa_obs_lat', 'date': 'noaa_obs_date'},
        inplace=True)
    reindexed_xbd_points.rename(
        columns={'geometry': 'xbd_obs_geometry', 'lon': 'xbd_obs_lon', 'lat': 'xbd_obs_lat'},
        inplace=True)

    joined_df = reindexed_xbd_points.join(reindexed_noaa_xbd_hurricanes, how='inner').set_index('xbd_index')
    joined_df.sort_values(by='xbd_index', inplace=True)
    df = calc_distance_between_df_cols(
        joined_df, [['noaa_obs_lat', 'noaa_obs_lon'], ['xbd_obs_lat', 'xbd_obs_lon']], 'shortest_distance_to_track')

    return df


def merge_dirs_of_grib_files_to_ncs(
    parent_dir_path: str
):
    """TODO"""
    print('todo')


def merge_grib_files_to_nc(
    grib_dir_path: str
):
    """TODO"""
    # load in all files in folder
    file_paths = '/'.join((grib_dir_path, '*.grib'))
    grib_dir_name = '/'.split(grib_dir_path)[-2]

    xa_dict = {}
    for file_path in tqdm.tqdm(glob.glob(file_paths)):
        # get name of file
        file_name = file_path.split('/')[-1]
        # read into xarray
        xa_dict[file_name] = xr.load_dataset(file_path, engine="cfgrib")

    # merge TODO: apparently conflicting values of 'step'. Unsure why.
    out = xr.merge([array for array in xa_dict.values()], compat='override')
    # save as new file
    nc_file_name = '.'.join((grib_dir_name, 'nc'))
    save_file_path = '/'.join((grib_dir_path, nc_file_name))
    out.to_netcdf(path=save_file_path)
    print(f'{nc_file_name} saved successfully')

from __future__ import annotations

import pandas as pd
import geopy
import shapely
from h3 import utils


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

    if 'geometry' in df.columns:
        # if geometry column not containing shapely Point objects
        if not type(df.geometry.iloc[0]) == shapely.geometry.point.Point:
            df.geometry = df.geometry.apply(
                lambda x: utils.geometry_ops.convert_point_string_to_point(x))
            # generate lat-lon columns from any Point objects
            df = generate_lat_lon_from_points_cols(df, ['geometry'])

    if set(['lat', 'lon']).issubset(df.columns):
        # make geometry column of shapely Point objects
        df = points_from_df_lat_lon_cols(df, point_col_name=new_point_col_name)
    # common variation
    elif set(['lat', 'long']).issubset(df.columns):
        df.rename(columns={'long': 'lon'}, inplace=True)
        df = points_from_df_lat_lon_cols(df, point_col_name=new_point_col_name)

    return df


def points_from_df_lat_lon_cols(
    df: pd.DataFrame,
    point_col_name: str = 'geometry'
) -> pd.DataFrame:
    """TODO: docstring"""
    df[point_col_name] = df.apply(lambda row: shapely.geometry.point.Point(
        row['lon'], row['lat']), axis=1)

    return df


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
    utils.simple_functions.checklistLengthsEqual([col_names, poi, buffer_val])

    for i, col in enumerate(col_names):
        if type(poi[i]) == pd.Timestamp:
            # specify the buffer as a Timestamp object (separating time and unit)
            buffer = pd.Timedelta(buffer_val[i][0], buffer_val[i][1])
        else:
            buffer = buffer_val[i]
        # restrict to only observations within the range
        df = df[df[col].between(poi[i]-buffer, poi[i]+buffer)]

    return df


def exclude_df_rows_by_range(
    df: pd.DataFrame,
    col_names: list[str],
    value_bounds: list[tuple[float]] | list[float],
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
    value_bounds : list[ [tuple[float], list[float]] ]
        e.g. [ (start_val1,end_val1), ..., (start_valN,end_valN) ]
        list of tuples (or lists) specifying minimum and maximum values to allow
    buffer : [ list[float], list[tuple[float,str]] ] = 0
        add buffer on either side of value_bounds. Defaults to no buffer. Useful
        for specifying weather station observations must exist some time before
        and after the event of interest

    Returns
    -------
    restricted pd.DataFrame object (sub-set of original df)
    """
    # check lists same lengths
    utils.simple_functions.checklistLengthsEqual([col_names, value_bounds])

    for i, col in enumerate(col_names):
        if type(value_bounds[i][0]) == pd.Timestamp:
            # specify the buffer as a Timestamp object (separating time and unit)
            buffer = pd.Timedelta(buffer[0], buffer[1])

        df = df[df[col].between(
            min(value_bounds[i])-buffer, max(value_bounds[i])+buffer)]

    return df


def concat_df_cols(
    df: pd.DataFrame,
    concatted_col_name: str,
    cols_to_concat: list[str],
    delimiter: str = ""
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
    delimiter : str, optional
        character to insert in between column values. Defaults to empty string

    Returns
    -------
    pd.DataFrame
        with additional concatted column
    """
    df[concatted_col_name] = df[cols_to_concat].astype(str).apply(
        delimiter.join, axis=1)

    return df


def generate_lat_lon_from_points_cols(
    df: pd.DataFrame,
    points_cols: list[str]
) -> None:
    """Generate a column(s) of lat and lon from column(s) of shapely.Point
    objects. Column(s) added to current df being processed

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


def calc_distance_between_df_cols(
    df: pd.DataFrame,
    cols_compare: list[tuple[str]] | list[list[str]],
    new_col_name: str = "distance",
) -> pd.DataFrame:
    """Calculate the geodesic distance between sets of lat/lon values.
    See https://geopy.readthedocs.io/en/stable/#module-geopy.distance for more info.

    Parameters
    ----------
    df: pd.DataFrame
        df containing two pairs of lat/lon values
    cols_compare: list[[tuple[str]] or list[list[str]]
        list of columns of lat/lon values. Inputted as pairs as a tuple or list
    new_col_name: str, optional
        The default is 'distance'.

    Returns
    -------
    pd.DataFrame
        copy of df with an extra 'distance' column
    """
    if not len(cols_compare) == 2:
        raise ValueError(
            'Cannot compare more or fewer than two sets of lat/lon values at a time')

    df[new_col_name] = df.apply(
        lambda x: geopy.distance.geodesic(
            (x[cols_compare[0][0]], x[cols_compare[0][1]]),
            (x[cols_compare[1][0]], x[cols_compare[1][1]])).km, axis=1)

    return df


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

    N.B. currently discarding any timezone information
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
    centre_coords : list[float] or tuple[float]
        geographical centre about which to restrict df
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
    if not set(['lat', 'lon']).issubset(df.columns):
        raise ValueError('Columns by name of lat and lon not found in df')

    # if choosing to find closest N points at any distance away
    if distance_buffer is None:
        # set arbitrarily small distance buffer
        distance_buffer = 1
        df_spatial_lim = exclude_df_rows_symmetrically_around_value(
            df, ['lat', 'lon'], centre_coords,
            [distance_buffer, distance_buffer])
        # expand distance buffer until minimum number reached
        while len(df_spatial_lim) <= min_number:
            distance_buffer += 1
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

    start, end = calculate_first_last_dates_from_df(
        df_noaa_weather_event, time_buffer)

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

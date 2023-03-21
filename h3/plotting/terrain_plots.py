from __future__ import annotations

import math
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio as rio
import richdem as rd
import pandas as pd

from pandas.core.groupby import DataFrameGroupBy
from typing import Literal

from h3 import logger
from h3.dataloading.terrain_ef import get_buildings_bounding_box, get_coastpoints_range, get_distance_coast
from h3.utils.directories import get_dem_dir
from h3.dataloading import general_df_utils


def plot_coastline(coast_points: list) -> None:
    """Function to plot the coastlines as point on a PlateCarree projection

    Parameters
    ----------
    coast_points : list
        a list of the coordinates of the coast. Given in `lat`, `lon`.
    """
    logger.info("Plotting the coastline")
    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    # Add a global map background
    ax.stock_img()
    # Plot the coast points
    lon = [c[1] for c in coast_points]
    lat = [c[0] for c in coast_points]
    ax.scatter(lon, lat, s=1, transform=ccrs.PlateCarree())
    # Set x-label and y-label
    ax.set_xlabel(r"Longitude (°)", fontsize=12)
    ax.set_ylabel(r"Latitude (°)", fontsize=12)
    # Set x-ticks and y-ticks
    xticks = np.arange(-180, 190, 20)
    yticks = np.arange(-90, 100, 20)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.set_title("Plot of the coastline points1")
    plt.show()


def plot_buildings_coast(building_groups, coast_points, dis_threshold: int = 2, dis_to_coast: bool = True) -> None:
    # assuming the building is not more than dis_threshold latitude and longitude away from the coast,

    coast_points = np.array(coast_points)
    n_cols = 3      # nbr of columns for the plot
    n_groups = len(building_groups)
    n_rows = math.ceil(n_groups / n_cols)
    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 12),
        dpi=300,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for i, (group_name, group_data) in enumerate(building_groups):
        east, north, west, south = get_buildings_bounding_box(group_data)
        # only select coast points that are in range (dis_threshold)
        points_within_range = get_coastpoints_range(
            bounding_box=(east, north, west, south),
            coast_points=coast_points,
            dis_threshold=dis_threshold
        )

        # plot the buildings and the coastline data that been chopped
        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        ax.set_xlim(west - dis_threshold, east + dis_threshold)
        ax.set_ylim(south - dis_threshold, north + dis_threshold)
        ax.add_feature(cfeature.LAND.with_scale("10m"))
        ax.add_feature(cfeature.OCEAN.with_scale("10m"))

        # Plot the coast points
        ax.scatter(points_within_range[:, 1], points_within_range[:, 0], s=5, transform=ccrs.PlateCarree())
        ax.scatter(group_data["lon"], group_data["lat"], s=5, transform=ccrs.PlateCarree())

        # Set x-label and y-label
        ax.set_xlabel("Longitude (°)", fontsize=12)
        ax.set_ylabel("Latitude (°)", fontsize=12)

        # Set x-ticks and y-ticks
        xticks = np.arange(west - dis_threshold, east + dis_threshold, dis_threshold)
        yticks = np.arange(south - dis_threshold, north + dis_threshold, dis_threshold)

        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())

        if dis_to_coast:
            buildings = np.column_stack((group_data["lat"], group_data["lon"]))
            # points_within_range are lat lon
            coast_within_range = np.column_stack((points_within_range[:, 0], points_within_range[:, 1]))
            nearest_coast_point, _ = get_distance_coast(buildings, coast_points=coast_within_range)

            ax.scatter(
                nearest_coast_point[:, 1],
                nearest_coast_point[:, 0],
                s=5,
                transform=ccrs.PlateCarree(),
                c="yellow"
            )
            ax.plot(
                [buildings[:, 1], nearest_coast_point[:, 1]],
                [buildings[:, 0], nearest_coast_point[:, 0]],
                c="k",
                linewidth=0.1
            )

    for i in range(n_cols * n_rows):
        fig.delaxes(axs.flatten()[i + 1])   # remove the *ax* from the figure; update the current Axes
    plt.show()


def building_plot(building_groups: DataFrameGroupBy, dis_threshold: int = 1):
    # plot the building locations for verification
    n_groups = len(building_groups)  # group number
    n_cols = 3  # column number
    n_rows = math.ceil(n_groups / n_cols)  # raw number

    fig, axs = plt.subplots(
        n_rows,
        n_cols,
        figsize=(12, 12),
        dpi=300,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    for i, (group_name, group_data) in enumerate(building_groups):
        east, north, west, south = get_buildings_bounding_box(group_data)

        row = i // n_cols
        col = i % n_cols
        ax = axs[row, col]
        ax.set_xlim(west - dis_threshold, east + dis_threshold)
        ax.set_ylim(south - dis_threshold, north + dis_threshold)

        ax.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.5)
        ax.add_feature(cfeature.LAND.with_scale("10m"))
        ax.add_feature(cfeature.OCEAN.with_scale("10m"))

        # plot the locations of buildings
        ax.scatter(group_data["lon"], group_data["lat"], s=5, transform=ccrs.PlateCarree(), c="orange")

        # Set x-label and y-label
        ax.set_xlabel("Longitude (°)", fontsize=12)
        ax.set_ylabel("Latitude (°)", fontsize=12)

        # Set x-ticks and y-ticks
        xticks = np.arange(west - dis_threshold, east + dis_threshold, dis_threshold)
        yticks = np.arange(south - dis_threshold, north + dis_threshold, dis_threshold)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())

    # while i < n_cols * n_rows - 1:
    for i in range(n_cols * n_rows):
        fig.delaxes(axs.flatten()[i + 1])

    plt.show()


def plot_dem(dem_urls: list) -> None:
    # list of the path of the extracted tif file
    dem_tif_path_list = [f"{os.path.basename(os.path.splitext(dem_file)[0])}_dem.tif" for dem_file in dem_urls]
    # Set the number of columns and rows for the plot
    num_cols = 3
    num_rows = -(-len(dem_tif_path_list) // num_cols)

    # Create a new figure with the appropriate number of subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20), dpi=300)

    # Iterate over the files and plot each one in a subplot
    for i, file in enumerate(dem_tif_path_list):
        row, col = divmod(i, num_cols)
        ax = axs[row, col]

        tif_path = os.path.join(get_dem_dir(), file)

        with rio.open(tif_path) as dem:
            dem_array = dem.read(1).astype("float64")
            handle = rio.plot.show(
                dem_array,
                transform=dem.transform,
                ax=ax,
                title=f"{file.split('_')[1]}",  # only taking the reference of the file
                cmap="gist_earth",
                vmin=0,
                vmax=np.percentile(dem_array, 99)
            )  # plot DEM map

            im = handle.get_images()[0]
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Elevation (m)")
            ax.set_xlabel("Longitude(°)")
            ax.set_ylabel("Latitude(°)")

    # Remove any unused subplots
    for i in range(len(axs.flat)):
        if i >= len(dem_tif_path_list):
            fig.delaxes(axs.flat[i])

    plt.show()


def plot_map_location(
    building_groups,
    dem_tif_path_list,
    dem_tif_short_name_list,
    terrain_attribute: Literal["dem", "slope", "aspect"]
):
    # plot dem map and building locations
    # Set the number of columns and rows for the plot
    num_cols = 3
    num_rows = -(-len(dem_tif_path_list) // num_cols)

    # Create a new figure with the appropriate number of subplots
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 20), dpi=300)

    for i, (group_name, group_data) in enumerate(building_groups):
        with rio.open(dem_tif_path_list[i]) as dem:
            dem_array = dem.read(1).astype("float64")
            crs = dem.crs
            transform = dem.transform

        row, col = divmod(i, num_cols)
        ax = axs[row, col]

        if terrain_attribute == "dem":
            data = dem_array
            cbar_label = "Elevation (m)"
            cmap = "gist_earth"
        elif terrain_attribute == "slope":
            dem4slope = rd.rdarray(dem_array, no_data=-9999)
            dem4slope.geotransform = [0, 1, 0, 0, 0, 1, 0]
            data = rd.TerrainAttribute(dem4slope, attrib="slope_riserun")
            cbar_label = "Slope (%)"
            cmap = "PuBu"
        elif terrain_attribute == "aspect":
            dem4slope = rd.rdarray(dem_array, no_data=-9999)
            dem4slope.geotransform = [0, 1, 0, 0, 0, 1, 0]
            data = rd.TerrainAttribute(dem4slope, attrib="aspect")  # calculate aspect
            cbar_label = "Aspect (°)"
            cmap = "twilight_shifted"

        handle = rio.plot.show(
            data,
            transform=transform,
            ax=ax,
            title=f"{dem_tif_short_name_list[i]}",
            cmap=cmap,
            vmin=0,
            vmax=np.percentile(dem_array, 99)
        )  # plot DEM map

        gdf = gpd.GeoDataFrame(
            group_data.copy(),
            geometry=gpd.points_from_xy(group_data.lon, group_data.lat),
            crs=crs
        )
        gdf.plot(ax=handle, color="red")  # plot location of buildings
        im = handle.get_images()[0]
        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label(cbar_label)
        ax.set_xlabel("Longitude(°)")
        ax.set_ylabel("Latitude(°)")

    # Remove any unused subplots
    for i in range(len(axs.flat)):
        if i >= len(dem_tif_path_list):
            fig.delaxes(axs.flat[i])

    plt.show()


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
                            subplot_kw={'projection': ccrs.PlateCarree()})

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
        axs[i].add_feature(cfeature.LAND.with_scale('10m'))
        axs[i].add_feature(cfeature.OCEAN.with_scale('10m'))

        gdf_noaa_hurricanes = gpd.GeoDataFrame(df_noaa_hurricanes[df_noaa_hurricanes['name'] == event_name])
        gdf_noaa_hurricanes.plot(ax=axs[i], color='blue', markersize=100, alpha=0.3, label='NOAA hurricane observation')

        if df_xbd_points is not None:
            gdf_xbd_points = gpd.GeoDataFrame(df_xbd_points[df_xbd_points['disaster_name'] == event_name])
            gdf_xbd_points.plot(ax=axs[i], color='red', markersize=0.5, alpha=1, label='xbd observation')
            av_lat, av_lon = general_df_utils.calc_means_df_cols(gdf_xbd_points, ['lat', 'lon'])

            # if limiting plot to a show at least 'min_number_weather' datapoints
            if min_number_weather:
                # ensure that at least 'min_number_weather' hurricane observation points are in the image
                gdf_noaa_hurricanes = general_df_utils.limit_df_spatial_range(
                    gdf_noaa_hurricanes, [av_lat, av_lon], min_number=min_number_weather)

            if df_stations is not None:
                gdf_stations = gpd.GeoDataFrame(
                    df_stations, geometry=gpd.points_from_xy(df_stations.lon, df_stations.lat), crs="EPSG:4326")
                # inactive weather stations
                gdf_stations_inactive = general_df_utils.station_availability(
                    gdf_stations, gdf_noaa_hurricanes, available=False)
                gdf_stations_inactive.plot(
                    ax=axs[i], color='darkorange', markersize=10, label='inactive weather station')
                # active weather stations
                gdf_stations_active = general_df_utils.station_availability(
                    gdf_stations, gdf_noaa_hurricanes, available=True)
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


def main():
    from h3.dataloading.terrain_ef import get_coastlines
    coast_points = get_coastlines()
    plot_coastline(coast_points=coast_points)


if __name__ == "__main__":
    main()

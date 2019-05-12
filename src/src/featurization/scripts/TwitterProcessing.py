""" A module for processing twitter data
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import time
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
from matplotlib import colors


# Contiguous states
states = gpd.read_file(os.path.join('..', 'resources', 'contiguous.geojson'))
statesExtent = np.array(states.cascaded_union.envelope.bounds).reshape(2,2).T


def getWordMask(
        df,
        words,
        verbose=0):
    """ Get a dict which maps words to masks that are
        True for tweets that contain the word """
    # Compute the mask of all words containing the element
    wordMasks = dict(zip(words, np.full((len(words), df.shape[0]), False)))
    for idx, tweetText in enumerate(df['TEXT'].values):
        if verbose > 0:
            if idx % 10000 == 0:
                print('RECORD ({0} / {1})'.format(idx, df.shape[0]))
        for word in words:
            try:
                wordMasks[word][idx] = (word == '') or (word in tweetText.split())
            except AttributeError:
                pass
    return wordMasks


def constructMapsKDE(
        df,
        words,
        extent=None,
        res = 1.,
        bandwidth=1.,
        atol=0,
        rtol=0,
        kernel='gaussian',
        algorithm='auto',
        useGeoBounds=False,
        verbose=0):
    """ Construct Kernel Density Estimation maps """
    # Get the extent, if not specified
    if extent is None:
        extent = statesExtent
    else:
        extent = np.asarray(extent)
    # Get masks for all words
    wordMasks = getWordMask(df, words, verbose=verbose)
    # Create grids
    lonStep = res
    latStep = res
    lons = np.arange(*extent[0], lonStep)
    lats = np.arange(*extent[1], latStep)
    lonGrid, latGrid = np.meshgrid(lons, lats)
    gridShape = lonGrid.shape
    flatGrid = np.vstack((lonGrid.flatten(), latGrid.flatten())).T
    # Generate a mask within the bounds of the US
    if useGeoBounds:
        if verbose > 0:
            print('Generating boxes')
        boxes = []
        for lon, lat in flatGrid:
            box = shapely.geometry.box(lon,
                                       lat,
                                       lon + lonStep,
                                       lat + latStep)
            boxes.append(box)
        boxes = gpd.GeoDataFrame(geometry=boxes)
        if verbose > 0:
            print('Computing intersection of boxes')
        intersectMask = boxes.intersects(states.cascaded_union).astype(bool).values
    else:
        intersectMask = np.full(flatGrid.shape[0], True)
    # For each word, generate the density map
    densityMaps = {}
    for word, wordMask in wordMasks.items():
        if wordMask.any():
            # Coordinates
            coords = df[['LON', 'LAT']][wordMask].values
            # Fit KDE
            kde = KernelDensity(bandwidth=bandwidth,
                                kernel=kernel,
                                atol=atol,
                                rtol=rtol,
                                algorithm=algorithm)
            if verbose > 0:
                ts = time.time()
                print('Fitting KDE for word: "{0}"'.format(word))
            kde.fit(coords)
            # Compute density map
            if verbose > 0:
                te = time.time()
                print(te - ts)
                ts = te
                print('Creating density map')
            densityMap = np.full(flatGrid.shape[0], -np.inf)
            densityMap[intersectMask] = kde.score_samples(flatGrid[intersectMask])
            densityMap[densityMap < np.log(min(atol, rtol))] = -np.inf
            densityMap = densityMap.reshape(*gridShape)
            densityMaps[word] = densityMap
            if verbose > 0:
                te = time.time()
                print(te - ts)
                print('Done')
    return densityMaps, lons, lats


def plotRelativeDensityMap(
        word,
        words,
        densityMaps,
        lons,
        lats):
    """ Plot a relative density map """
    hmap = (np.mean(np.exp(densityMaps[words == word]), axis=0) -
            np.mean(np.exp(densityMaps[words == '']), axis=0))
    lonGrid, latGrid = np.meshgrid(lons, lats)
    states.plot(color='gray')
    plt.pcolormesh(lonGrid,
                   latGrid,
                   hmap,
                   cmap='RdBu',
                   vmin=-1,
                   vmax=+1,
                   norm=colors.SymLogNorm(linthresh=1e-3),
                   alpha=0.5)
    plt.tight_layout()
    plt.show()
    return

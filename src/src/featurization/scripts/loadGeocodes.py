import pandas as pd
import geopandas as geopd
import numpy as np
import shapely
import geopy
import sys


if __name__ == '__main__':
    dstFile = sys.argv[1]
    srcFile = sys.argv[2]

    # Load the dataframe
    locDf = pd.read_csv(srcFile)

    # Load the world shape
    world = geopd.read_file(geopd.datasets.get_path('naturalearth_lowres'))
    usa = world[world['name'] == 'United States']
    citySeries = geopd.GeoDataFrame(geometry=locDf[['LON', 'LAT']].T.apply(shapely.ops.Point))

    try:
        #mask = locDf.isna().any(axis=1)
        mask = ~citySeries.intersects(usa.buffer(1.0).cascaded_union)
        for idx in np.where(mask)[0]:
            locName = locDf.iloc[idx]['LOC']
            print('({1} / {2}): "{0}"'.format(locName, idx, locDf.shape[0]))
            tryCount = 0
            locPt = None
            while tryCount < 5 and locPt is None:
                tryCount += 1
                try:
                    # Get the geocode for this file
                    locPt = geopd.tools.geocode([locName+', USA'], provider='arcgis')
                    # Get information
                    addr = locPt.address[0]
                    lon = locPt.geometry[0].coords.xy[0][0]
                    lat = locPt.geometry[0].coords.xy[1][0]
                    # Store in dataframe
                    locDf['Addr'][idx] = addr
                    locDf['LON'][idx] = lon
                    locDf['LAT'][idx] = lat
                except geopy.exc.GeocoderTimedOut:
                    print('Timeout ({0})'.format(tryCount))
    except:
        print('Cleaning up: Saving to {0}'.format(dstFile))
    with open(dstFile, 'w') as wfp:
        print(locDf.to_csv(index=False), file=wfp)

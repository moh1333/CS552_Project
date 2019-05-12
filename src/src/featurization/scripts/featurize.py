""" Featurization script for dataset.
    USAGE:
        python3 <tweetFile> <wordFile> <densityMapFile>
"""

import sys
import os
import TwitterProcessing as tp
import numpy as np
import pandas as pd


if __name__ == '__main__':
    # Command line args
    tweetFilePath = sys.argv[1]
    wordFilePath = sys.argv[2]
    densityMapFile = sys.argv[3]

    # Load the words from the word file
    print('Reading words')
    words = []
    with open(wordFilePath, 'r') as rfp:
        for line in rfp:
            words.append(line.strip())
    #if '' not in words:
    #    words = [''] + words

    # Load all of the tweets
    print('Loading tweet dataframe')
    df = pd.read_csv(tweetFilePath)

    # Compute density maps
    print('Computing tweet density maps')
    densityMaps, lons, lats = tp.constructMapsKDE(df,
                                                  words,
                                                  res=0.25,
                                                  bandwidth=0.75,
                                                  atol=1e-6,
                                                  rtol=1e-6,
                                                  kernel='gaussian',
                                                  useGeoBounds=True,
                                                  verbose=1)

    # Save the density maps
    print('Saving to file')
    densityMapsArr = np.array([densityMaps[word] for word in words])
    np.savez(densityMapFile,
             densityMaps=densityMaps,
             words=words,
             lons=lons,
             lats=lats)


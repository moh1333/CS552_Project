""" Load data from the Cheng-Caverlee-Lee Sep. 2009
    Twitter scrape.
"""
import pandas as pd
import geopandas as geopd
import numpy as np
import re


def loadTweetsInAFormatThatMakesSense(inFile):
    print('Loading data file')
    with open(inFile, 'r') as rfp:
        data = rfp.read()
    # Split into lines
    print('Finding all records')
    lineExp = r'\d+\t\d+\t.*\t\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}[\r\n]'
    lines = re.findall(lineExp, data)
    # Now split each line
    print('Parsing records')
    rows = []
    nLines = len(lines)
    lineTick = nLines // 100
    for idx, line in enumerate(lines):
        if ((idx + 1) % lineTick) == 0:
            print('Progress: {0}'.format(100 * idx // nLines))
        # Split the line into tab-separated values
        sp = line.strip().split('\t')
        row = [sp[0], sp[1], '\t'.join(sp[2:-1]), sp[-1]]
        rows.append(row)
    # Put it all in a dataframe
    print('Writing it all to a dataframe')
    df = pd.DataFrame(rows, columns=['USERID', 'TWEETID', 'TEXT', 'TIMESTAMP'])
    return df


def combineUserInfo(tweetfile, userfile):
    print('Loading user dataframe')
    userdf = pd.read_csv(userfile).drop_duplicates(subset='USERID', keep='first').set_index('USERID', drop=True)

    with open(tweetfile, 'r') as rfp:
        lines = rfp.readlines()
    rows = []
    nLines = len(lines)
    lineTick = nLines // 100
    for idx, line in enumerate(lines):
        if ((idx + 1) % lineTick) == 0:
            print('Progress: {0}'.format(100 * idx // nLines))
        userid = int(line.split(',')[1)
    
    return lines

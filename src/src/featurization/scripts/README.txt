This is the code for the featurization module.
To create a feature file, run the featurize script:

python3 featurize.py <preprocessed tweet file> <words txt file> <save file>

This will save the results in a .npz file, which contains 4 variables:
    words: Words corresponding to each density map
    densityMaps: log-probability density maps, one per word
    lons: x-axis edge of each density map
    lats: y-axis edge of each density map

For our analysis, we used "training_set_contiguous_clean.csv" as the preprocessed tweet file.

Plots can be made interactively using the TwitterProcessing.plotRelativeDensityMap function.

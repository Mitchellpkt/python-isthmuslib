#########################
## Imports
#########################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


##############################
## Scatter plot
##############################

# LEGACY CODE - NEEDS REFACTOR
def scatter(xData, yData, xlabel='', ylabel='', title='', xlim=None, ylim=None, figsize=None, facecolor='w',
            xylabelsize=15, titlesize=20, xscale='linear', yscale='linear', markersize=3, markercolor='green',
            grid=False, legend=None, markerstyle='o', lines=False, linestyle='-', linewidth=None,
            rollingMedianBinWidth=None, rollingMeanBinWidth=None, linecolor=None, plotBestFit=False,
            watermarkText=None, watermarkPlacement=None, watermarkSize=10, watermarkColor='grey'):
    # Process defaults
    if figsize is None:
        figsize = (13, 7)
    if str(type(yData)) != "<class 'dict'>":
        yDataCache = yData
        yData = dict()
        yData[''] = yDataCache
        hadLabel = False
    else:
        hadLabel = True

    # Initialize
    legendStrings = list()
    legendHandles = list()
    keys = list(yData.keys())

    # Create the plot
    f = plt.figure(figsize=figsize, facecolor=facecolor)
    for traceIndex in range(len(keys)):
        # This loop is over y data sets, if a dictionary was provided
        thisKey = keys[traceIndex]
        yDataSet = yData[thisKey]
        if lines:
            plt.plot(xData, yDataSet, color=markercolor, linestyle=linestyle, linewidth=linewidth)
        trace = plt.scatter(xData, yDataSet, s=markersize, c=markercolor, marker=markerstyle)
        if hadLabel:
            legendHandles.append(trace)
            legendStrings.append(thisKey)

        if (rollingMeanBinWidth is not None) or (rollingMedianBinWidth is not None):
            temporaryDataFrame = pd.DataFrame()
            temporaryDataFrame['xData'] = xData
            temporaryDataFrame['yData'] = yDataSet
            temporaryDataFrame.sort_values(by='xData', ascending=True, inplace=True)

            if rollingMeanBinWidth is not None:
                trace = plt.scatter(temporaryDataFrame.xData,
                                    temporaryDataFrame['yData'].rolling(rollingMeanBinWidth).mean(),
                                    color=linecolor, linestyle=linestyle, linewidth=linewidth)
                plt.plot(temporaryDataFrame.xData, temporaryDataFrame['yData'].rolling(rollingMeanBinWidth).mean(),
                         color='k', linestyle=linestyle, linewidth=linewidth)
                thisString = thisKey + ' (rolling mean, bin width = ' + str(rollingMeanBinWidth) + ')'
                legendHandles.append(trace)
                legendStrings.append(thisString)

            if rollingMedianBinWidth is not None:
                trace = plt.scatter(temporaryDataFrame.xData,
                                    temporaryDataFrame['yData'].rolling(rollingMedianBinWidth).median(),
                                    color=linecolor, linestyle=linestyle, linewidth=linewidth)
                plt.plot(temporaryDataFrame.xData, temporaryDataFrame['yData'].rolling(rollingMedianBinWidth).median(),
                         color='k', linestyle=linestyle, linewidth=linewidth)
                thisString = thisKey + ' (rolling median, bin width = ' + str(rollingMedianBinWidth) + ')'
                legendHandles.append(trace)
                legendStrings.append(thisString)

        if plotBestFit:
            m, b = np.polyfit(xData, yDataSet, 1)
            bestFitY = [m * x + b for x in xData]
            plt.plot(xData, bestFitY, color='k')
            # Next few lines disabled until plot handles are handled
            # trace = plt.plot(xData, bestFitY)
            # thisString = thisKey + ' (best fit)'
            # legendHandles.append(trace)
            # legendStrings.append(thisString)

    # Bells and whistles
    plt.xlabel(xlabel, size=xylabelsize)
    plt.ylabel(ylabel, size=xylabelsize)
    plt.title(title, size=titlesize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if grid:
        plt.grid(grid)
    if legend:
        plt.legend(legendHandles, legendStrings)
    if watermarkText is not None:
        if watermarkPlacement is None:
            watermarkPlacement = (0.01, 0.97)
        xLimits = plt.xlim()
        yLimits = plt.ylim()
        xCoordinate = xLimits[0] + watermarkPlacement[0] * (xLimits[1] - xLimits[0])
        yCoordinate = yLimits[0] + watermarkPlacement[1] * (yLimits[1] - yLimits[0])
        plt.text(xCoordinate, yCoordinate, watermarkText, fontsize=watermarkSize, c=watermarkColor)
    return f


##############################
## Scatter plot (dictionary input)
##############################

# LEGACY CODE - NEEDS REFACTOR
def scatterDictionary(dataDict, xlabel='', ylabel='', title='', xlim=None, ylim=None, figsize=None, facecolor='w',
                      xylabelsize=15, titlesize=20, xscale='linear', yscale='linear', grid=False, legend=None,
                      lines=False, linestyle='-', linewidth=None, watermarkText=None, watermarkPlacement=None,
                      watermarkSize=10, watermarkColor='grey'):
    # Process defaults
    if figsize is None:
        figsize = (13, 7)

    # Initialize
    legendStrings = list()
    legendHandles = list()

    # Create the plot
    f = plt.figure(figsize=figsize, facecolor=facecolor)

    # Loop over data sets to process & plot
    for traceIndex in range(len(dataDict)):
        thisDict = dataDict[traceIndex]
        theseKeys = list(thisDict.keys())
        if ('x' in theseKeys) and ('y' in theseKeys):
            x = thisDict['x']
            y = thisDict['y']
        else:
            raise NameError('isthmuslib.scatterDict::NeedBothXandYdata')
        if 'markersize' in theseKeys:
            markersize = thisDict['markersize']
        else:
            markersize = None
        if 'markercolor' in theseKeys:
            markercolor = thisDict['markercolor']
        else:
            markercolor = None
        if 'markerstyle' in theseKeys:
            markerstyle = thisDict['markerstyle']
        else:
            markerstyle = None
        if 'label' in theseKeys:
            legendStrings.append(thisDict['label'])
        else:
            legendStrings.append('')

        # Add this data set to the plot
        if lines:
            plt.plot(x, y, color=markercolor, linestyle=linestyle, linewidth=linewidth)
        trace = plt.scatter(x, y, s=markersize, c=markercolor, marker=markerstyle)
        legendHandles.append(trace)

    # Bells and whistles
    plt.xlabel(xlabel, size=xylabelsize)
    plt.ylabel(ylabel, size=xylabelsize)
    plt.title(title, size=titlesize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if grid:
        plt.grid(grid)
    if legend:
        plt.legend(legendHandles, legendStrings)
    if watermarkText is not None:
        if watermarkPlacement is None:
            watermarkPlacement = (0.01, 0.97)
        xLimits = plt.xlim()
        yLimits = plt.ylim()
        xCoordinate = xLimits[0] + watermarkPlacement[0] * (xLimits[1] - xLimits[0])
        yCoordinate = yLimits[0] + watermarkPlacement[1] * (yLimits[1] - yLimits[0])
        plt.text(xCoordinate, yCoordinate, watermarkText, fontsize=watermarkSize, c=watermarkColor)
    return f


##############################
## Histogram plot
##############################

# LEGACY CODE - NEEDS REFACTOR
def hist(data, xlabel='', ylabel='frequency', title='', xlim=None, ylim=None, figsize=None, facecolor='w',
         xylabelsize=15, titlesize=20, xscale='linear', yscale='linear', color=None, bins=150, grid=False, legend=None,
         cumulative=False, density=False, internalFaceColor=None, alpha=None, watermarkText=None,
         watermarkPlacement=None, watermarkSize=10, watermarkColor='grey'):
    if figsize is None:
        figsize = (13, 7)

    # Process defaults
    if str(type(data)) != "<class 'dict'>":
        dataCache = data
        data = dict()
        data[''] = dataCache
        if color is None:
            color = 'green'

    # If the x-axis scale is 'log' this must be taken into account when picking bin edges!
    # (this is not the case with log y-axis, which is an easy transformation)
    if xscale == 'log':
        if type(bins) == int:
            bins = np.logspace(np.log10(min(data)), np.log10(max(data)), bins)
        else:
            bins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
            # This does NOT know how to handle negative numbers ^^

    # Initialize
    legendStrings = list()
    legendHandles = list()
    keys = list(data.keys())

    # Create the plot
    f, ax = plt.subplots(figsize=figsize, facecolor=facecolor)
    if internalFaceColor is not None:
        ax.set_facecolor(internalFaceColor)
    for traceIndex in range(len(keys)):
        thisKey = keys[traceIndex]
        legendStrings.append(thisKey)
        trace = plt.hist(data[thisKey], color=color, bins=bins, cumulative=cumulative, density=density, alpha=alpha)
        legendHandles.append(trace)

    # Bells and whistles
    plt.xlabel(xlabel, size=xylabelsize)
    plt.ylabel(ylabel, size=xylabelsize)
    plt.title(title, size=titlesize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if grid:
        plt.grid(grid)
    if legend:
        plt.legend(legendHandles, legendStrings)
    if watermarkText is not None:
        if watermarkPlacement is None:
            watermarkPlacement = (0.01, 0.97)
        xLimits = plt.xlim()
        yLimits = plt.ylim()
        xCoordinate = xLimits[0] + watermarkPlacement[0] * (xLimits[1] - xLimits[0])
        yCoordinate = yLimits[0] + watermarkPlacement[1] * (yLimits[1] - yLimits[0])
        plt.text(xCoordinate, yCoordinate, watermarkText, fontsize=watermarkSize, c=watermarkColor)
    return f


##############################
## 2D histogram plot
## Input: X & Y vectors
## Calculates: Z = counts
##############################

# LEGACY CODE - NEEDS REFACTOR
def hist2d(xData, yData, xlabel='', ylabel='frequency', title='', xlim=None, ylim=None, figsize=None,
           facecolor='w', xylabelsize=15, titlesize=20, xscale='linear', yscale='linear', cmap='jet', bins=50,
           grid=None, range=None, density=False, cmin=None, cmax=None, norm=None, bookends=1, watermarkText=None,
           watermarkPlacement=None, watermarkSize=10, watermarkColor='grey'):
    # Process defaults
    if figsize is None:
        figsize = (13, 7)
    if norm == 'log':
        # norm = colors.LogNorm(vmin=min(counts), vmax=max(counts))
        print('to-do: port log zscale from MATLAB isthmuslib code')
    if bookends != 1:
        print('to-do: port bookends functionality from MATLAB isthmuslib')

    # Create the plot
    f = plt.figure(figsize=figsize, facecolor=facecolor)
    plt.hist2d(xData, yData, bins=bins, range=range, density=density, cmin=cmin, cmax=cmax, cmap=plt.get_cmap(cmap),
               norm=norm)

    # Bells and whistles
    plt.xlabel(xlabel, size=xylabelsize)
    plt.ylabel(ylabel, size=xylabelsize)
    plt.title(title, size=titlesize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if grid:
        plt.grid(grid)
    if watermarkText is not None:
        if watermarkPlacement is None:
            watermarkPlacement = (0.01, 0.97)
        xLimits = plt.xlim()
        yLimits = plt.ylim()
        xCoordinate = xLimits[0] + watermarkPlacement[0] * (xLimits[1] - xLimits[0])
        yCoordinate = yLimits[0] + watermarkPlacement[1] * (yLimits[1] - yLimits[0])
        plt.text(xCoordinate, yCoordinate, watermarkText, fontsize=watermarkSize, c=watermarkColor)
    return f


##############################
## 2D histogram plot
## Input: X & Y & Z vectors
##############################

# LEGACY CODE - NEEDS REFACTOR
def heatmapDataFrame(df, xcol='x', ycol='y', zcol='z', xlabel='', ylabel='', annot=False, figsize=None,
                     facecolor='white', title='', xlim=None, ylim=None, vmin=None, vmax=None, xylabelsize=15,
                     titlesize=20, linewidth=None, cmap=None, cbar=True, mask=None, center=None, robust=None,
                     linecolor=None, watermarkText=None, watermarkPlacement=None, watermarkSize=10,
                     watermarkColor='grey'):
    # Process defaults
    if figsize is None:
        figsize = (13, 7)

    x = df[xcol]
    y = df[ycol]
    z = df[zcol]
    return heatmap(x, y, z, annot=annot, xlabel=xlabel, ylabel=ylabel, figsize=figsize,
                   facecolor=facecolor, title=title, xlim=xlim, ylim=ylim, vmin=vmin, vmax=vmax,
                   xylabelsize=xylabelsize, titlesize=titlesize, linewidth=linewidth, cmap=cmap, cbar=cbar, mask=mask,
                   center=center, robust=robust, linecolor=linecolor, watermarkSize=watermarkSize,
                   watermarkText=watermarkText, watermarkPlacement=watermarkPlacement)


# LEGACY CODE - NEEDS REFACTOR
def heatmap(x, y, z, xlabel='x', ylabel='y', annot=False, figsize=None, facecolor='white', title='',
            xlim=None, ylim=None, vmin=None, vmax=None, xylabelsize=15, titlesize=20, linewidth=None, cmap=None,
            cbar=True, mask=None, center=None, robust=None, linecolor=None, watermarkText=None,
            watermarkPlacement=None, watermarkSize=10, watermarkColor='grey'):
    # Process defaults
    if figsize is None:
        figsize = (13, 7)

    # Initial data wrangling
    df = pd.DataFrame()
    df['x'] = x
    df['y'] = y
    df['z'] = z
    dfPivoted = df.pivot("y", "x", "z")

    # Make the plot
    f = plt.figure(figsize=figsize, facecolor=facecolor)
    sns.heatmap(dfPivoted, annot=annot, vmin=vmin, vmax=vmax, linewidth=linewidth, cmap=cmap, cbar=cbar, mask=mask,
                center=center, robust=robust, linecolor=linecolor)

    # Bells and whistles
    plt.xlabel(xlabel, size=xylabelsize)
    plt.ylabel(ylabel, size=xylabelsize)
    plt.title(title, size=titlesize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if watermarkText is not None:
        if watermarkPlacement is None:
            watermarkPlacement = (0.01, 0.97)
        xLimits = plt.xlim()
        yLimits = plt.ylim()
        xCoordinate = xLimits[0] + watermarkPlacement[0] * (xLimits[1] - xLimits[0])
        yCoordinate = yLimits[0] + watermarkPlacement[1] * (yLimits[1] - yLimits[0])
        plt.text(xCoordinate, yCoordinate, watermarkText, fontsize=watermarkSize, c=watermarkColor)
    return f


##############################
## DataViz shotgun plots
##############################
# To do - log x-axis might not work

# LEGACY CODE - NEEDS REFACTOR
def shotgunPlots1d(df, useFields=None, xscale='linear', yscale='linear', figsize=(13, 7), cumulative=False,
                   density=False):
    if useFields == None:
        useFields = list(df.keys())
        # If there are non-numeric fields, things will get wonky

    for fIndex in range(len(useFields)):
        thisField = useFields[fIndex]
        thisData = df[thisField]
        try:
            hist(thisData, xlabel=thisField, title=thisField, xscale=xscale, yscale=yscale, figsize=figsize,
                 cumuative=cumulative, density=density)
        except:
            pass


# LEGACY CODE - NEEDS REFACTOR
def shotgunPlots2d(df, xColName, useFields=None, xscale='linear', yscale='linear', figsize=(13, 7),
                   types=['scatter', 'hist2d']):
    if useFields == None:
        useFields = list(df.keys())
        # If there are non-numeric fields, things will get wonky

    for fIndex in range(len(useFields)):
        thisField = useFields[fIndex]
        thisData = df[thisField]

        if any(atype == 'scatter' for atype in types):
            try:
                scatter(df[xColName], thisData, xlabel=xColName, ylabel=thisField, title=thisField, xscale=xscale,
                        yscale=yscale, figsize=figsize)
            except:
                pass

        if any(atype == 'hist2d' for atype in types):
            try:
                hist2d(df[xColName], thisData, xlabel=xColName, ylabel=thisField, title=thisField, xscale=xscale,
                       yscale=yscale, figsize=figsize)
            except:
                pass

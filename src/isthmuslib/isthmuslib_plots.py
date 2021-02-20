#########################
## Imports
#########################

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


##############################
## Scatter plot
##############################

def scatter(xData, yData, xlabel='', ylabel='', title='', xlim=None, ylim=None, figsize=None, facecolor='w',
            xylabelsize=15, titesize=20, xscale='linear', yscale='linear', markersize=3, markercolor='green',
            grid=False, legend=None, markerstyle='o', lines=False, linestyle='-', linewidth=None):
    # Process defaults
    if figsize is None:
        figsize = (13, 7)
    if str(type(yData)) != "<class 'dict'>":
        yDataCache = yData
        yData = dict()
        yData[''] = yDataCache

    # Initialize
    legendStrings = list()
    legendHandles = list()
    keys = list(yData.keys())

    # Create the plot
    f = plt.figure(figsize=figsize, facecolor=facecolor)
    for traceIndex in range(len(keys)):
        thisKey = keys[traceIndex]
        legendStrings.append(thisKey)
        if lines:
            plt.plot(xData, yData[thisKey], color=markercolor, linestyle=linestyle, linewidth=linewidth)
        trace = plt.scatter(xData, yData[thisKey], s=markersize, c=markercolor, marker=markerstyle)
        legendHandles.append(trace)

    # Bells and whistles
    plt.xlabel(xlabel, size=xylabelsize)
    plt.ylabel(ylabel, size=xylabelsize)
    plt.title(title, size=titesize)
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
    return f


##############################
## Scatter plot (dictionary input)
##############################

def scatterDictionary(dataDict, xlabel='', ylabel='', title='', xlim=None, ylim=None, figsize=None, facecolor='w',
                      xylabelsize=15, titesize=20, xscale='linear', yscale='linear', grid=False, legend=None,
                      lines=False, linestyle='-', linewidth=None):
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
    plt.title(title, size=titesize)
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
    return f


##############################
## Histogram plot
##############################

def hist(data, xlabel='', ylabel='frequency', title='', xlim=None, ylim=None, figsize=None, facecolor='w',
         xylabelsize=15, titesize=20, xscale='linear', yscale='linear', color=None, bins=150, grid=False, legend=None,
         cumulative=False):
    if figsize is None:
        figsize = (13, 7)

    # Process defaults
    if str(type(data)) != "<class 'dict'>":
        dataCache = data
        data = dict()
        data[''] = dataCache
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
    f = plt.figure(figsize=figsize, facecolor=facecolor)
    for traceIndex in range(len(keys)):
        thisKey = keys[traceIndex]
        legendStrings.append(thisKey)
        trace = plt.hist(data[thisKey], color=color, bins=bins, cumulative=cumulative)
        legendHandles.append(trace)

    # Bells and whistles
    plt.xlabel(xlabel, size=xylabelsize)
    plt.ylabel(ylabel, size=xylabelsize)
    plt.title(title, size=titesize)
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
    return f


##############################
## Heatmap plot
##############################

def hist2d(xData, yData, xlabel='', ylabel='frequency', title='', xlim=None, ylim=None, figsize=None,
           facecolor='w', xylabelsize=15, titesize=20, xscale='linear', yscale='linear', cmap='jet', bins=50,
           grid=None, range=None, density=False, cmin=None, cmax=None, norm=None, bookends=1):
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
    plt.title(title, size=titesize)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    plt.xscale(xscale)
    plt.yscale(yscale)
    if grid:
        plt.grid(grid)
    return f


##############################
## DataViz shotgun plots
##############################
# To do - log x-axis might not work

def shotgunPlots1d(df, useFields=None, xscale='linear', yscale='linear', figsize=(13, 7), cumulative=False):
    if useFields == None:
        useFields = list(df.keys())
        # If there are non-numeric fields, things will get wonky

    for fIndex in range(len(useFields)):
        thisField = useFields[fIndex]
        thisData = df[thisField]
        try:
            hist(thisData, xlabel=thisField, title=thisField, xscale=xscale, yscale=yscale, figsize=figsize,
                 cumuative=cumulative)
        except:
            pass


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

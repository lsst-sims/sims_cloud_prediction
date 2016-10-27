from __future__ import division
from __future__ import print_function

import numpy as np
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from astropy.time import Time

import sys
from datetime import date

from cloudServer import CloudServer
import cloudMap
from cloudMap import CloudMap

from astropy.io import fits
import os

# This annoying "registering dimensions" is necessary because I haven't
# split FITS management into a different class. Given that the final
# system will be dealing with live camera data probably in the form of
# healpix maps, I haven't bothered to make a FitsManagement class.
# (I wouldn't have to do this if I just recalculated orderedIndices
# each time through fits2Hpix, but that would be very expensive)
areDimensionsRegistered = False
fitsWidth = 0
fitsHeight = 0
x = y = f = r = phi = theta = orderedIndices = None

def registerFitsDimensions(width, height):
    # this method has two purposes: ensure that all fits files being considered
    # have the same dimensions, and precalculate theta and orderedIndices for
    # use in fits2Hpix so we don't have to recalculate them each time
    # Sorry this is kind of ugly with globals and such -- see comment above
    global areDimensionsRegistered, fitsWidth, fitsHeight
    if areDimensionsRegistered:
        if width != fitsWidth or height != fitsHeight:
            raise ValueError("New fits width/height don't match existing")
        else:
            return

    # if we get here, this is the first time registering fits dimensions
    fitsWidth = width
    fitsHeight = height
    areDimensionsRegistered = True

    xCent = fitsWidth / 2
    yCent = fitsHeight / 2

    global f, x, y, r, phi, theta, orderedIndices
    # TODO I'm not totally sure if this is the correct way to calculate
    # the effective focal length. For ut111515, the value I got from
    # Chris Stubbs was 852
    f = np.sqrt(xCent**2 + yCent**2) / 2 # effective focal length in pixels

    # cartesian coordinates in the focal plane
    y, x = np.ogrid[-yCent:yCent, -xCent:xCent]

    # (r,phi) are polar coordinates in the focal plane
    r = np.sqrt(y**2 + x**2)
    phi = np.arctan2(y, x)

    # theta is the zenith angle
    theta = 2 * np.arcsin(r / (2 * f))

    # this is a list of indices in the order that pixels appear
    # in the fits files
    orderedIndices = hp.ang2pix(cloudMap.nside, theta, phi)

def fits2Hpix(fitsData, bias):
    """ Convert a fits image to a healpix map

    @returns    a healpix map with the fits data in it
    @param      fitsData: a fits image array
    @param      bias: bias from the fits header

    This function maps a fits allsky image into a healpix map, cutting
    the fits image off at a maximum zenith angle.
    """

    fitsData -= bias

    # TODO make this faster: cut off zenith angle in orderedIndices
    # instead of here
    fitsData[np.where(theta > cloudMap.thetaMax)] = -1

    hpix = np.zeros(cloudMap.npix)
    hpix[orderedIndices] = fitsData
    
    return hpix

def calcAccuracy(predMap, trueMap): 
    """ Calculate various forms of accuracy

    @returns    a 3-tuple containing the fraction of pixels:
                 -that were actually cloudy that were predicted to be cloudy 
                 -that were actually cloudy that were predicted to be clear  
                 -that were actually clear  that were predicted to be cloudy 
    @param      predMap: a CloudMap representing the predicted cloud coverage
    @param      trueMap: a CloudMap representing the true cloud coverage
    
    """

    # TODO cloudyThreshold would presumably be determined by the tolerance
    # LSST has for looking through clouds. I don't know that tolerance
    # so I've arbitrarily set it
    cloudyThreshold = 1000
    numTrueCloudy = np.size(np.where(trueMap > cloudyThreshold)[0])
    numTrueClear  = np.size(np.where(trueMap < cloudyThreshold)[0])

    if numTrueCloudy == 0:
        fracCloudyandCloudy = 0
        fracPredClearAndTrueCloudy = 0
    else:
        fracCloudyandCloudy = np.size(
            np.where((predMap>cloudyThreshold) & (trueMap>cloudyThreshold))[0]
        ) / numTrueCloudy

        #print("Of the pixels which turned out to be cloudy, ",
        #      fracCloudyandCloudy * 100,
        #      "percent of them were predicted to be cloudy.")

        fracPredClearAndTrueCloudy = np.size(
            np.where((predMap<cloudyThreshold) & (trueMap>cloudyThreshold))[0]
        ) / numTrueCloudy

        #print("Of the pixels which turned out to be cloudy,",
        #      fracPredClearAndTrueCloudy * 100,
        #      "percent of them were predicted to be clear.")

    if numTrueClear == 0:
        fracPredCloudyAndTrueClear = 0
    else:
        fracPredCloudyAndTrueClear = np.size(
            np.where((predMap>cloudyThreshold) & (trueMap<cloudyThreshold))[0]
        ) / numTrueClear

        #print("Of the pixels which turned out to be clear,",
        #      fracPredCloudyAndTrueClear * 100,
        #      "percent of them were predicted to be cloudy.")
        
    return (fracCloudyandCloudy, 
            fracPredClearAndTrueCloudy, 
            fracPredCloudyAndTrueClear)

if __name__ == "__main__":
    # the argument to this script is the date of the fits files to be converted
    if len(sys.argv) != 2:
        exit("usage is testPredClouds.py date")
    predDate = sys.argv[1]

    # TODO update dir
    # the files for each date are stored in a subdirectory of /data/allsky
    #dataDir = "/data/allsky/ut" + predDate + "/fits/"
    dataDir = "/home/drothchi/"
    filePrefix = "ut" + predDate + ".daycal."
    filePostfix = ".fits"
    def getFilename(filenum):
        return dataDir + filePrefix + str(filenum).zfill(4) + filePostfix

    # these numbers (inclusive) specify the range of files to convert
    fileNumStart = 201
    fileNumEnd = 333
    fileNums = range(fileNumStart, fileNumEnd + 1)

    # first get mjds for each fits file
    mjds = np.zeros(len(fileNums)).astype(float)
    for i in range(len(fileNums)):
        fileName = getFilename(fileNums[i])
        if os.path.exists(fileName):
            mjd = fits.open(fileName)[0].header["mjd-obs"]
        else:
            mjd = -1
        mjds[i] = mjd

    # start up the cloud server
    cloudServer = CloudServer()

    # put placeholder zeros in each figure to specify the imshow settings
    # this is probably the wrong way of doing this but it doesn't 
    # particularly matter
    placeholder = np.zeros((cloudMap.xyMax, cloudMap.xyMax))

    # I think this sets the physical size of the output
    plt.figsize=(10,10)

    # create a 3x3 grid of subplots
    fig, axarr = pylab.subplots(4,3)
    imgs = np.array([[None for x in range(3)] for y in range(4)])

    # and put the placeholder image where images will later be placed
    for y in range(3):
        for x in range(3):
            axarr[y,x].axis("off")
            if y == 0 and x != 1:
                # top left and right corners don't get images
                continue
            imgs[y,x] = axarr[y,x].imshow(placeholder, vmax=4000, cmap=plt.cm.jet)

    # put titles on each subplot
    axarr[0,1].set_title("True Clouds")

    # put the stats legend in upper left
    axarr[0,0].axis("off")
    axarr[0,0].text(0, 0.2, "% cloudy pred as cloudy", color="blue")
    axarr[0,0].text(0, 0.5, "% cloudy pred as clear", color="green")
    axarr[0,0].text(0, 0.8, "% clear pred as cloudy", color="red")

    # the second row contains the prediction images
    axarr[1,0].set_title("~5 Min Prediction")
    axarr[1,1].set_title("~10 Min Prediction")
    axarr[1,2].set_title("~20 Min Prediction")

    # the third row contains the difference images
    axarr[2,0].set_title("Diff from True")
    axarr[2,1].set_title("Diff from True")
    axarr[2,2].set_title("Diff from True")

    # the third row contains the accuracy charts
    for i in range(3):
        axarr[3,i].set_title("Accuracy")
        axarr[3,i].set_ylim([0,100])
        axarr[3,i].set_xticklabels([])

    # keep track of all the predictions in predMaps
    # make predictions 5, 10, and 20 minutes into the future
    predTimes = [m / 60 / 24 for m in [5, 10, 20]]
    predMaps = [[None for i in range(len(predTimes))] 
                for j in range(len(fileNums))]

    # we keep track of 3 kinds of accuracy for len(predTimes) prediction times
    accuracies = np.zeros((len(fileNums), len(predTimes), 3))

    # loop through each image, predicting 5, 10, and 20 minutes ahead
    # at each step and storing the results in the arrays above
    for i in range(len(fileNums)):
        curFileNum = fileNums[i]
        curMjd = mjds[i]

        fileName = getFilename(curFileNum)
        if not os.path.exists(fileName):
            # there are some missing files. Too bad
            # TODO copy prediction image to get rid of lag
            continue


        # get a CloudMap representation of this fileNum
        fitsFile = fits.open(fileName)[0]
        (width, height) = (fitsFile.header["naxis1"], fitsFile.header["naxis2"])
        registerFitsDimensions(width, height)
        fitsData = fitsFile.data.astype(float)
        bias = fitsFile.header["bias"]

        hpix = fits2Hpix(fitsData, bias)
        curMap = cloudMap.fromHpix(str(curFileNum), hpix)

        # and post to the CloudServer
        cloudServer.postCloudMap(curMjd, curMap)

        if not cloudServer.isReadyForPrediction():
            # can't predict before we have enough posted to the server
            continue

        # make a prediction for each of the times we're predicting ahead
        for predTimeId in range(len(predTimes)):
            # predTime is the amount of time ahead we're predicting
            predTime = predTimes[predTimeId]

            # figure out which fileNum we're trying to predict
            # this will be the fileNum approximately predTime days after
            # the ith fileNum
            predFileNumId = -1
            for j in range(i, len(fileNums)):
                # yes this is linear search through a sorted array
                # not sure why I did this -- pretty sure there's some
                # built-in function that would work?? 
                # not like perf in this loop matters...
                if mjds[j] > curMjd + predTime:
                    predFileNumId = j
                    break
            if predFileNumId != -1:
                predMap = cloudServer.predCloudMap(mjds[predFileNumId])
                predMaps[predFileNumId][predTimeId] = predMap


        # now plot all the images for number i
        # TODO should keep cloudData private--can probably override subtraction
        # actually this might be very annoying to do...

        # set the true cloud map at the current time
        imgs[0,1].set_data(curMap.cloudData)

        # list the time of the true map in the upper right slot
        t = Time(curMjd, format="mjd").datetime
        axarr[0,2].cla()
        axarr[0,2].axis("off")
        axarr[0,2].text(0.2, 0.5, date.strftime(t, "%H:%M:%S"))

        # set the pred image, diff image, and accuracy image for each predTime
        for predTimeId in range(len(predTimes)):
            predMap = predMaps[i][predTimeId]
            if predMap is not None:
                imgs[1,predTimeId].set_data(predMap.cloudData)
                diff = np.abs(predMap.cloudData - curMap.cloudData)
                diff[np.where(~predMap.validMask | ~curMap.validMask)] = 0
                imgs[2,predTimeId].set_data(diff)

                accuracies[i][predTimeId] = list(calcAccuracy(predMap, curMap))

                axarr[3,predTimeId].cla()
                axarr[3,predTimeId].set_title("Accuracy")
                axarr[3,predTimeId].set_ylim([0,100])
                axarr[3,predTimeId].set_xticklabels([])
                axarr[3,predTimeId].plot(accuracies[i-10:i,predTimeId,:] * 100)

        print("saving number", curFileNum)
        path = "fullpngs/ut" + predDate
        if not os.path.isdir(path):
            os.mkdir(path)
        pylab.savefig(path + "/" + str(curFileNum).zfill(5) + ".png", dpi=200)


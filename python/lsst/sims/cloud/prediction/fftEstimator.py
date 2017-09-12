from __future__ import division
from __future__ import print_function

""" The code in this file was adapted from code written by Ian Sullivan 
    (thanks Ian!) """

import copy
import numpy as np
from skimage.feature import register_translation
from scipy.signal import fftconvolve
from scipy import ndimage
import matplotlib.pyplot as plt

from cloudStateEstimator import CloudStateEstimator
from cloudMap import CloudMap
import cloudMap
from cloudMap import xyMax
from cloudState import CloudState


class FftEstimator(CloudStateEstimator):
    @staticmethod
    def _doEstimateCloudState(map1, map2, deltaT):
        """Calculate the x,y shift in pixels between pairs of images"""

        # Needs to be calculated from the image! 
        # Maybe set to a value that will use a set fraction of the pixels
        threshold = 6e4  

        # This should be based on physics or observations of cloud velocity.
        maxShift = 20

        maps = [map1, map2]
        ffts = []
        for cloudMap in maps:
            mask = cloudMap.validMask
            image = cloudMap.cloudData

            fillFraction = np.mean(mask)

            # calculate the raw ffts of the mask and cloud map
            maskFft = np.fft.fft2(mask)
            imageFft = np.fft.fft2(image)

            # scale the mask fft
            scale = imageFft[0, 0] / maskFft[0, 0]
            maskFft *= scale

            # save the normalized fft
            normalized = (np.fft.fft2(image) - maskFft) / fillFraction
            ffts.append(normalized)


        peakImageSize = 2 * maxShift + 1
        velocityPeakImage = np.zeros((peakImageSize, peakImageSize))

        diff = ffts[0] - ffts[1]
        # normalize pixels in diff which are above threshold 
        aboveThreshold = abs(diff) > threshold
        diff[aboveThreshold] = diff[aboveThreshold] / ffts[0][aboveThreshold]
        # set pixels in diff below threshold to 0
        diff[~aboveThreshold] = 0.0

        # only look for shifts less than maxShift
        velocityImage = np.fft.fftshift(np.real(np.fft.fft2(diff)))
        minSearch = xyMax / 2 - maxShift
        maxSearch = xyMax / 2 + maxShift + 1
        velocityImage = velocityImage[minSearch:maxSearch, minSearch:maxSearch]
        # subtract out the mean
        velocityImage -= np.mean(velocityImage)

        # TODO could add the velocityImage from many consecutive pairs for 
        # better and faster velocity estimation
        # here I just use the single velocityImage from the passed-in pair
        velocityPeakImage += velocityImage

        posImage = np.zeros((2 * maxShift + 1, 2 * maxShift + 1))
        negImage = np.zeros((2 * maxShift + 1, 2 * maxShift + 1))
        posImage[velocityPeakImage > 0] = velocityPeakImage[velocityPeakImage > 0]
        negImage[velocityPeakImage < 0] = -velocityPeakImage[velocityPeakImage < 0]
        centroidPositiveY, centroidPositiveX = FftEstimator._naiveCentroid(posImage)
        centroidNegativeY, centroidNegativeX = FftEstimator._naiveCentroid(negImage)
        shiftY = centroidPositiveY - centroidNegativeY
        shiftX = centroidPositiveX - centroidNegativeX

        cloudVelocity = (shiftY / deltaT, shiftX / deltaT)
        return CloudState(vel=cloudVelocity)


    @staticmethod
    def _naiveCentroid(image, center=None, sigma=1.):
        """Simplistic centroid calculation."""
        if center is None:
            center = np.unravel_index(image.argmax(), image.shape)
        ySize, xSize = image.shape
        yInds, xInds = np.meshgrid(np.arange(ySize), np.arange(xSize), indexing='ij')
        gaussFilter = np.exp(-((xInds - center[1])**2 + (yInds - center[0])**2) / (2 * sigma**2))
        return(ndimage.measurements.center_of_mass(image * gaussFilter))

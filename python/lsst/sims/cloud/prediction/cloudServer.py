from __future__ import division
from __future__ import print_function

import numpy as np
from lsst.sims.cloud.prediction.cloudState import CloudState
from lsst.sims.cloud.prediction.rmseEstimator import RmseEstimator


__all__ = ['CloudServer', 'CachedMap']


class CloudServer:

    def __init__(self, frame_gap=1):
        """
        Paramters
        ---------
        frame_gap : int (1)
            The gap between frames to calculate velocity. (number)
        """
        # throw out stale cloud maps once we reach more than this many
        self._MAX_CACHED_MAPS = 40
        # calculate velocity vectors between frames this far apart
        # XXX---uuuuuggggghhhh. This should be a time gap I think.
        self._NUM_VEL_CALC_FRAMES = frame_gap

        self._cachedMaps = []
        self._cachedRmses = {}

        # use this estimator for estimating cloud states
        # self.estimator = FftEstimator
        self.estimator = RmseEstimator

    def isReadyForPrediction(self):
        return len(self._cachedMaps) > self._NUM_VEL_CALC_FRAMES

    def postCloudMap(self, mjd, cloudMap):
        """ Notify CloudServer that a new cloud map is available

        @returns    void
        @param      mjd: the time the image was taken
        @param      cloudMap: the cloud cover map
        @throws     ValueError if the caller attempts to post cloud maps
                    out of chronological order
        """

        if len(self._cachedMaps) > 0:
            if mjd <= self._cachedMaps[-1].mjd:
                raise ValueError("cloud maps must be posted in order of mjd")

        self._cachedMaps.append(CachedMap(mjd, cloudMap))
        if len(self._cachedMaps) > self._MAX_CACHED_MAPS:
            self._cachedMaps.pop(0)

    def predCloudMap(self, mjd):
        """ Predict the cloud map

        @returns    a CloudMap instance with the predicted cloud cover
        @param      mjd: the time the prediction is requested for
        @throws     RuntimeWarning if not enough cloud maps have been posted
        @throws     ValueError if mjd is before the latest posted cloud map
        """

        numMaps = len(self._cachedMaps)
        if numMaps <= self._NUM_VEL_CALC_FRAMES:
            raise RuntimeWarning("too few clouds have been posted to predict")

        latestMap = self._cachedMaps[-1].cloudMap
        latestMjd = self._cachedMaps[-1].mjd
        if mjd <= latestMjd:
            raise ValueError("can't predict the past")

        # calculate cloudState for all pairs that are the desired gap appart.
        # XXX-TOO. Do we want to do the full matrix of the i,j-th velocity calculation?
        for i in range(self._NUM_VEL_CALC_FRAMES, numMaps):
            if self._cachedMaps[i].cloudState is None:
                cachedMap1 = self._cachedMaps[i - self._NUM_VEL_CALC_FRAMES]
                cachedMap2 = self._cachedMaps[i]
                deltaT = cachedMap2.mjd - cachedMap1.mjd
                self._cachedMaps[i].cloudState = self.estimator.estimateCloudState(cachedMap1.cloudMap,
                                                                                   cachedMap2.cloudMap,
                                                                                   deltaT)

        vs = [cachedMap.cloudState.vel
              for cachedMap in self._cachedMaps[self._NUM_VEL_CALC_FRAMES:]]
        v = np.median(vs, axis=0)

        predMap = latestMap.transform(CloudState(vel=v), mjd - latestMjd)
        return predMap


class CachedMap:
    """ Wrapper class for parameters describing the clouds' dynamical state """
    def __init__(self, mjd, cloudMap, cloudState = None):
        self.mjd = mjd
        self.cloudMap = cloudMap
        self.cloudState = cloudState

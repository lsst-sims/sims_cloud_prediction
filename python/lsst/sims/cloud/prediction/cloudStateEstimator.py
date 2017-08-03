from __future__ import division

from cloudMap import CloudMap
import abc

class CloudStateEstimator(object):
    #__metaclass__ = abc.ABCMeta

    @staticmethod
    def _doEstimateCloudState(map1, map2, deltaT):
        """ Method to actually do the state estimation. 
        
        This method should never be called directly. Instead,
        estimateCloudState() should be called, which does error handling
        and then calls this method.

        I would make this an abstract method but it doesn't seem like abstract
        static methods are supported until Python 3
        """
        raise NotImplementedError("subclass did not implement " +
                                  "_doEstimateCloudState()")

    @classmethod
    def estimateCloudState(cls, map1, map2, deltaT):
        """ Find the cloud state using two closely-spaced CloudMap objects

        Use scipy.optimize.minimize to find the velocity vector which minimizes
        the rmse between map1 and map2 when translated by the velocity.

        @returns    A best guess of the current dynamical CloudState
        @param      map1: a CloudMap of the clouds at some time
        @param      map2: a CloudMap of the clouds at a time deltaT after map1
        @param      deltaT: the difference in mjd between map1 and map2
        @throws     ValueError if deltaT <= 0
        @throws     TypeError if map1 or map2 are not CloudMap objects
        """

        if deltaT <= 0:
            raise ValueError("deltaT must be >= 0")
        if not isinstance(map1, CloudMap):
            raise TypeError("map1 must be a CloudMap instance")
        if not isinstance(map2, CloudMap):
            raise TypeError("map2 must be a CloudMap instance")

        return cls._doEstimateCloudState(map1, map2, deltaT)

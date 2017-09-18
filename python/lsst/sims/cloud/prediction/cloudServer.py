from __future__ import division
from __future__ import print_function
import numpy as np
from lsst.sims.cloud.prediction import CloudMap

__all__ = ['CloudServer']


class CloudServer:
    """Assume we can model the clouds as a single pattern traveling at a constant velocity.
    """

    def __init__(self, velocity_time_max=20., velocity_time_min=0.5, predict_time_max=20.):
        """
        Paramters
        ---------
        velocity_time_max : float (10.)
            Ignore frames older than this when computing velocity of a frame
        velocity_time_min : float (0.5)
            Don't try to compute velocities between frames if the time between them is
            less than this (default 0.5 mintues)
        predict_time_max : float (10.)
            Ignore cloudmaps older than this when predicting the future (default 10 minutes)
        """
        # throw out stale cloud maps once we reach more than this many
        self.predict_time_max = predict_time_max/24./60.
        # calculate velocity vectors between frames this far apart
        self.velocity_time_min = velocity_time_min/24./60.
        self.velocity_time_max = velocity_time_max/24./60.

        # The max time we need to keep any frame cached
        self.max_time = np.max([self.predict_time_max, self.velocity_time_max])

        self._cachedMaps = []
        self._cachedMJDs = []
        # Hold the nside of the maps
        self.nside = None

    def isReadyForPrediction(self):
        # Check that we have a velocity in the latest map
        return self._cachedMaps[-1].vel is not None

    def prune_cache(self):
        # delta times in min
        delta_t = (self._cachedMJDs[-1] - np.array(self._cachedMJDs))
        too_old = np.where(delta_t > self.max_time)[0]
        for indx in too_old:
            self._cachedMaps.pop(0)
            self._cachedMJDs.pop(0)

    def postCloudMap(self, cloudMap):
        """ Notify CloudServer that a new cloud map is available, use cached maps
        to compute a velocity if needed.

        Parameters
        ----------
        cloudMap : cloudMap
            A new cloudMap object to add to the internal cache
        """

        if self.nside is None:
            self.nside = cloudMap.nside
        elif self.nside != cloudMap.nside:
            raise ValueError("cloud maps must be same nside resolution")

        if len(self._cachedMaps) > 0:
            if cloudMap.mjd <= self._cachedMaps[-1].mjd:
                raise ValueError("cloud maps must be posted in order of mjd")

        self._cachedMaps.append(cloudMap)
        self._cachedMJDs.append(cloudMap.mjd)
        self.prune_cache()

        # Compute a velocity for the new frame if needed
        # XXX-Note, once velocity is calculated, I don't bother updating. I suppose
        # one could update a cloudMaps velocity based on later frames. Whatevs.
        if self._cachedMaps[-1].vel is None:
            delta_t = (self._cachedMJDs[-1] - np.array(self._cachedMJDs))
            matching_maps = np.where((delta_t < self.velocity_time_max) &
                                     (delta_t > self.velocity_time_min))[0]
            if np.size(matching_maps) > 0:
                vels = []
                for indx in matching_maps:
                    vels.append(self._cachedMaps[-1].vrelmap(self._cachedMaps[indx]))
                final_velocity = np.median(vels, axis=0)
                self._cachedMaps[-1].set_vel(final_velocity)

    def predCloudMap(self, mjd, mjd_now=None):
        """ Predict the cloud map.

        Parameters
        ----------
        mdj : float
            The modified Julian Date to predict
        mjd_now : float (None)
            The current MJD, uses the last map in the cache if None

        Returns
        -------
        cloudMap object

        """

        if not self.isReadyForPrediction():
            raise RuntimeWarning("Have not been able to compute a velocity yet")

        latestMjd = self._cachedMJDs[-1]
        if mjd <= latestMjd:
            raise ValueError("can't predict the past")

        if mjd_now is None:
            mjd_now = self._cachedMJDs[-1]

        # Find which maps are good for propigating
        delta_t = mjd_now - np.array(self._cachedMJDs)
        indxs = np.where(delta_t < self.predict_time_max)[0]

        predicted_maps = []

        nmaps = 0.
        for indx in indxs:
            if self._cachedMaps[indx].vel is not None:
                predicted_maps.append(self._cachedMaps[indx].transform(mjd).cloudData)
                nmaps += 1.

        predicted_maps = np.array(predicted_maps)

        # use MJD=-1 to make sure this object doesn't get confused with a real cloud map
        predicted_map = CloudMap(np.sum(predicted_maps, axis=0) / nmaps, -1)

        return predicted_map


from __future__ import division
from __future__ import print_function

import numpy as np
import healpy as hp
from scipy.signal import convolve2d
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab


__all__ = ['CloudMap', 'toHpix', 'fromHpix', 'nside2cloudConfig', 'cloudmap2nside']


def nside2cloudConfig(nside=32, zenith_distance_max=70., height_scale=3.,
                      sunAvoidRadius=30.):
    """Hold default parameters about resolution of cloud mask model

        Parameters
        ----------
        nside : int (32)
            The HEALpixel nside to use for sky maps
        zenith_distance_max : float (70.)
            The maximum zenith distance in degrees to compute the map. Default
            of 70 degrees ~ 2.9 airmass (80 degrees would be X=5.8)
        height_scale : float (3.)
            The height of the cloud layer relative to the cloud plane size.
            (I think this just scales the velocities up and down?)
        sunAvoidRadius : float (30.)
            Avoid area around the sun with this radius (pixels)

        """
    result = {}
    result['nside'] = nside
    result['npix'] = hp.nside2npix(nside)

    # ignore pixels in healpix maps with theta > thetaMax
    result['thetaMax'] = np.radians(zenith_distance_max)

    # use an XY plane with approximately somewhat more pixels than there
    # are in the hpix so we don't lose all resolution at high theta
    result['xyMax'] = int(np.sqrt(result['npix'])) * 2
    result['xyCent'] = int(result['xyMax'] / 2)

    # when we convert hpix to cartesian, the resulting xy map is square
    # but the signal is only found in a circle inscribed in the square
    # (since the passed-in healpixes don't go all the way to the horizon)
    # rMax is the radius in pixels of that circle
    result['rMax'] = 0.9 * result['xyMax'] / 2

    # z is the vertical distance in pixels from the observer to the clouds
    # It is chosen to make the skymap fill our XY coordinates.
    cloud_height = result['xyMax']/2./height_scale
    result['z'] = cloud_height

    # minimum distance from the sun in pixels
    result['sunAvoidRadius'] = sunAvoidRadius

    # y and x are useful for making masks
    result['y'], result['x'] = np.ogrid[0:result['xyMax'], 0:result['xyMax']]

    # maintain a mask of pixels within rMax
    result['insideRMaxMask'] = (result['y'] - result['xyCent'])**2 + (result['x'] - result['xyCent'])**2 <= result['rMax']**2

    return result


class CloudMap:
    """ Cartesian representation of the cloud cover

    Instance variables:
    mapId:      a unique identifier for the map
    cloudData:  a numpy array of shape (xyMax, xyMax) mapping the cloud cover
    sunPos:     the position [y,x] of the sun in this image
    validMask:  a binary mask indicating which pixels in the image are valid

    Methods:
    __init__(data, sunPos): constructor
    isPixelValid(point):    return whether point is part of the cloud map
    getSunPos():            find the sun in this cloud map
    transform(cS, time):    transform the cloud map according to cloud state
                            cS over a duration time
    __getitem__((y,x)):     allows for syntax like cloudMap[y,x]
    __eq__(self, other):    allows for syntax like map == m5threshold
    other magic cmp methods...
    plot(maxPixel, title):  plot the cloud map
    max():                  calculate the maximum pixel value
    std():                  calculate the std dev of the valid pixels
    mean():                 calculate the mean of the valid pixels
    """

    def __init__(self, cloudData, mjd, sunPos = None, maskSun=False,
                 fftpad=20, vel=None):
        """ Initialize the CartesianSky

        Parameters
        ----------
        cloudData : np.array
            cloud cover pixel values in an x-y plane.
        sunPos : ? (None)
            Position of the sun in cloudData. Calculates the sun position if None is passed
            in and then calculates the valid mask for the sky map.
        cloud_config : cloudConfig instance (None)
            The configuration object, the default is loaded if set to None.
        fftpad : int (20)
            The number of pixels to use when computing the FFT of the cloud frame
        vel : np.array
            The x,y velocity of the clouds in this frame in pix/day.
        """

        self.mjd = mjd
        nside = cloudmap2nside(cloudData)
        self.nside = nside
        self.cc = nside2cloudConfig(nside=self.nside)
        if cloudData.shape != (self.cc['xyMax'], self.cc['xyMax']):
            raise ValueError("the passed in cloud data has the wrong shape")
        if sunPos is not None and (sunPos[0] < 0 or sunPos[0] > self.cc['xyMax'] or
                                   sunPos[1] < 0 or sunPos[1] > self.cc['xyMax']):
            print("the passed-in sunPos is invalid:", sunPos)
            # TODO sometimes the sun position is invalid since it got shifted
            # off the map due to the velocity propagation. This should
            # probably be handled better
            sunPos = None

        self.cloudData = cloudData
        # allow the caller to pass in a sunPos if it's already known
        if sunPos is None:
            self.sunPos = self.getSunPos()
        else:
            self.sunPos = sunPos
        if maskSun:
            # keep track of which pixels are valid
            sunY, sunX = self.sunPos
            outsideSunMask = (self.cc['y'] - sunY)**2 + (self.cc['x'] - sunX)**2 >= self.cc['sunAvoidRadius']**2
            self.validMask = self.cc['insideRMaxMask'] & outsideSunMask
        else:
            self.validMask = self.cc['insideRMaxMask']

        self.cloudData[np.logical_not(self.validMask)] = -1
        self.fftpad = fftpad

        self.addFFT()
        self.vel = vel

    def set_vel(self, vel):
        """Set the velocity of the cloud
        """
        self.vel = vel

    def addFFT(self):
        """Compute the FFT of the cloudData
        """
        temp_cloud = self.cloudData.copy()
        # Set mask pixels to zero
        temp_cloud[np.where(temp_cloud == -1)] = 0
        temp_cloud = np.pad(temp_cloud, self.fftpad, 'constant', constant_values=0)
        self.cloudfft = np.fft.fft2(temp_cloud)

    def vrelmap(self, cloud0):
        """Find the velocity in pix/day between another frame
        Let's make cloud0 an earlier frame
        """
        mult = self.cloudfft * np.conjugate(cloud0.cloudfft)
        # Cross-correlation of the two cloud planes
        cc = np.fft.ifft2(mult)
        i, j = np.unravel_index(cc.argmax(), cc.shape)
        deltaT = self.mjd - cloud0.mjd
        vel = np.array([i, j])/deltaT
        return vel

    def isPixelValid(self, point):
        """ Return whether the pixel is a valid part of the sky map

        @returns    True if the pixel is valid, False otherwise
        @param      point: the pixel in question
        @throws     ValueError if point is outside the image

        A pixel is invalid if it's outside of rMax or too close to the sun.
        """
        # this check approximately doubles the total time of this method
        # this function is (or was) about 15% of total execution time
        # if point[0] < 0 or point[1] < 0 or point[0] > xyMax or point[1] > xyMax:
        #    raise ValueError("the supplied point:", point,
        #                     "is outside the sky map")

        # doing mask[p[0],p[1]] is about 50% faster than mask[tuple(p)]
        # this function is (or was) about 15% of total execution time
        return self.validMask[point[0], point[1]]

    def __getitem__(self, args):
        # given a CartesianSky object cart, this method allows other
        # code to use cart[y,x] instead of having to do cart.cart[y,x]
        # which would breach abstraction anyway
        (y, x) = args
        return self.cloudData[y, x]

    def getSunPos(self):
        """ Find the position of the sun in the image

        @returns    a point [y,x] indicating the sun's position
                    within self.cloudData

        This method smooths out the sky map and then finds the maximum
        pixel value in the smoothed image. If the smoothed image has
        multiple pixels which share the same maximum, this chooses
        the first such pixel.
        """
        # average the image to find the sun
        n = 10
        k = np.ones((n, n)) / n**2
        avg = convolve2d(self.cloudData, k, mode="same")

        sunPos = np.unravel_index(avg.argmax(), avg.shape)
        return sunPos

    def transform(self, mjd):
        """ Transform our cloud map according to

        Parameters
        ----------
        vel : np.array
            A two-element array with the x,y velocity in pix/day
        time : float
            Amount of time to advance in days.
        mId : int or str (None)
            The ID to assing to the returned map

        Returns
        -------
        CloudMap object at requested time.

        """

        time = mjd - self.mjd

        direction = np.round(np.array(self.vel) * time).astype(int)

        # translate the array by padding it with zeros and then cropping off the
        # extra numbers
        if direction[0] >= 0:
            padY = (direction[0], 0)
        else:
            padY = (0, -1 * direction[0])
        if direction[1] >= 0:
            padX = (direction[1], 0)
        else:
            padX = (0, -1 * direction[1])

        paddedData = np.pad(self.cloudData, (padY, padX),
                            mode="constant", constant_values=-1)

        # if spreading was added to cloudState, might want to deal with that
        # somewhere around here

        # now crop paddedData to the original size
        cropY = (padY[1], paddedData.shape[0] - padY[0])
        cropX = (padX[1], paddedData.shape[1] - padX[0])
        transformedData = paddedData[cropY[0]:cropY[1], cropX[0]:cropX[1]]

        # replace all pixels which are -1 with 0.
        # XXX--need to make clear these are unknown pixels, clouds could blow in
        (invalidY, invalidX) = np.where((transformedData == -1) &
                                        (self.cloudData != -1))
        transformedData[invalidY, invalidX] = 0

        return CloudMap(transformedData, self.mjd+time, sunPos = self.sunPos + direction)

    def plot(self, maxPixel, title=""):
        plt.figure(title)
        pylab.imshow(self.cloudData, vmax = maxPixel, cmap=plt.cm.jet)
        plt.colorbar()

    def max(self):
        return np.max(self.cloudData)

    def std(self):
        return np.std(self.cloudData[self.validMask])

    def mean(self):
        return np.mean(self.cloudData[self.validMask])


def fromHpix(hpix, mjd=0.):
    """ Convert a healpix image to a cartesian cloud map

    @returns    a CloudMap object with the data from the hpix
    @param      hpix: the healpix to be converted

    The top plane in the crude picture below is the cartesian plane
    where the clouds live. The dome is the healpix that we're
    looking "through" to see the clouds
    _________________
          ___
         /   \
        |  o  |

    To find out which healpix pixel corresponds to (x,y), we convert
    (x,y) to (r,phi). Then, we figure out which theta corresponds to
    the calculated r.
    """

    # now for each (x,y), sample the corresponding hpix pixel
    # see fits2Hpix() for an explanation of x, y, and cart

    nside = hp.npix2nside(hpix.size)

    cc = nside2cloudConfig(nside=nside)

    x = np.repeat([np.arange(-cc['xyCent'], cc['xyCent'])], cc['xyMax'], axis=0).T
    y = np.repeat([np.arange(-cc['xyCent'], cc['xyCent'])], cc['xyMax'], axis=0)
    cart = np.swapaxes([y, x], 0, 2)

    # calculate theta and phi of each pixel in the cartesian map
    r = np.linalg.norm(cart, axis=2)
    phi = np.arctan2(y, x).T
    theta = np.arctan(r / cc['z'])

    # ipixes is an array of pixel indices corresponding to theta and phi
    ipixes = hp.ang2pix(nside, theta, phi)

    # move back from physical coordinates to array indices
    y += cc['xyCent']
    x += cc['xyCent']
    y = y.astype(int)
    x = x.astype(int)

    # set the cloud data pixels to the corresponding hpix pixels
    cloudData = np.zeros((cc['xyMax'], cc['xyMax']))
    cloudData[y.flatten(), x.flatten()] = hpix[ipixes.flatten()]

    return CloudMap(cloudData, mjd)


def cloudmap2nside(cloudData):
    xyMax = cloudData.shape[0]
    npix_approx = int((xyMax/2.)**2)
    nside_approx = (npix_approx/12.)**0.5
    nside = int(np.round(nside_approx))
    return nside


def toHpix(cloudMap):
    """ Convert a CloudMap to a healpix image

    @returns    a healpix image of the clouds
    @param      cloudMap: a CloudMap with the cloud cover data

    For each pixel in hpix, sample from the corresponding pixel in cloudMap
    """
    nside = cloudmap2nside(cloudMap.cloudData)
    cc = nside2cloudConfig(nside)

    hpix = np.zeros(cc['npix'])
    (theta, phi) = hp.pix2ang(cc['nside'], np.arange(cc['npix']))

    r = np.tan(theta) * cc['z']
    x = np.floor(r * np.cos(phi)).astype(int)
    y = np.floor(r * np.sin(phi)).astype(int)

    # ignore all pixels with zenith angle higher than thetaMax
    x = x[theta < cc['thetaMax']]
    y = y[theta < cc['thetaMax']]
    ipixes = np.arange(cc['npix'])[theta < cc['thetaMax']]

    hpix[ipixes] = cloudMap[x + cc['xyCent'], y + cc['xyCent']]

    return hpix

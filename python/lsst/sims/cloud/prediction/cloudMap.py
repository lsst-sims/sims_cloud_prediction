from __future__ import division
from __future__ import print_function

import numpy as np
import healpy as hp

from scipy.signal import convolve2d

import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

nside = 32
npix = hp.nside2npix(nside)

# ignore pixels in healpix maps with theta > thetaMax
thetaMax = 70 * np.pi / 180

# use an XY plane with approximately somewhat more pixels than there 
# are in the hpix so we don't lose all resolution at high theta
xyMax = int(np.sqrt(npix)) * 2
xyCent = int(xyMax / 2)

# when we convert hpix to cartesian, the resulting xy map is square
# but the signal is only found in a circle inscribed in the square
# (since the passed-in healpixes don't go all the way to the horizon)
# rMax is the radius in pixels of that circle
rMax = 0.9 * xyMax / 2

# z is the vertical distance in pixels from the observer to the clouds
# It is chosen to make the skymap fill our XY coordinates.
z = 40

# minimum distance from the sun in pixels 
sunAvoidRadius = 30

# y and x are useful for making masks
y, x = np.ogrid[0:xyMax,0:xyMax]

# maintain a mask of pixels within rMax
insideRMaxMask = (y - xyCent)**2 + (x - xyCent)**2 <= rMax**2


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

    def __init__(self, mapId, cloudData, sunPos = None):
        """ Initialize the CartesianSky

        @returns    void
        @param      mapId: a unique identifier for the map (this is used
                    to avoid hashing the entire np.array upon a call to
                    hash(). If the map doesn't need to be hashable then
                    mapId can go away
        @param      cloudData: a np.array with the cloud cover pixel values
        @param      sunPos (optional): the position of the sun in the image
                    If not passed in, it will be calculated
        @throws     ValueError if cloudData has the wrong shape or sunPos is
                    outside of the image

        Calculates the sun position if None is passed in and then calculates
        the valid mask for the sky map.

        """
        self.mapId = mapId
        if cloudData.shape != (xyMax, xyMax):
            raise ValueError("the passed in cloud data has the wrong shape")
        if sunPos is not None and (sunPos[0] < 0 or sunPos[0] > xyMax or
                                   sunPos[1] < 0 or sunPos[1] > xyMax):
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

        # keep track of which pixels are valid
        sunY, sunX = self.sunPos
        outsideSunMask = (y - sunY)**2 + (x - sunX)**2 >= sunAvoidRadius**2

        self.validMask = insideRMaxMask & outsideSunMask

        self.cloudData[np.logical_not(self.validMask)] = -1

    def isPixelValid(self, point):
        """ Return whether the pixel is a valid part of the sky map

        @returns    True if the pixel is valid, False otherwise
        @param      point: the pixel in question
        @throws     ValueError if point is outside the image

        A pixel is invalid if it's outside of rMax or too close to the sun.
        """
        # this check approximately doubles the total time of this method
        # this function is (or was) about 15% of total execution time
        #if point[0] < 0 or point[1] < 0 or point[0] > xyMax or point[1] > xyMax:
        #    raise ValueError("the supplied point:", point, 
        #                     "is outside the sky map")

        # doing mask[p[0],p[1]] is about 50% faster than mask[tuple(p)]
        # this function is (or was) about 15% of total execution time
        return self.validMask[point[0],point[1]]

    def __getitem__(self, args):
        # given a CartesianSky object cart, this method allows other
        # code to use cart[y,x] instead of having to do cart.cart[y,x]
        # which would breach abstraction anyway
        (y,x) = args
        return self.cloudData[y,x]
    
    def hash(self):
        return hash(self.mapId)

    # Comparison methods, allowing for syntax like map1 > map2
    # and map3 <= 100

    def __eq__(self, other): 
        if isinstance(other, CloudMap):
            return self.cloudData.__eq__(other.cloudData)
        else:
            return self.cloudData.__eq__(other)

    def __ne__(self, other): 
        if isinstance(other, CloudMap):
            return self.cloudData.__ne__(other.cloudData)
        else:
            return self.cloudData.__ne__(other)

    def __lt__(self, other): 
        if isinstance(other, CloudMap):
            return self.cloudData.__lt__(other.cloudData)
        else:
            return self.cloudData.__lt__(other)

    def __gt__(self, other): 
        if isinstance(other, CloudMap):
            return self.cloudData.__gt__(other.cloudData)
        else:
            return self.cloudData.__gt__(other)

    def __le__(self, other): 
        if isinstance(other, CloudMap):
            return self.cloudData.__le__(other.cloudData)
        else:
            return self.cloudData.__le__(other)

    def __ge__(self, other): 
        if isinstance(other, CloudMap):
            return self.cloudData.__ge__(other.cloudData)
        else:
            return self.cloudData.__ge__(other)

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
        k = np.ones((n,n)) / n**2
        avg = convolve2d(self.cloudData, k, mode="same")

        sunPos = np.unravel_index(avg.argmax(), avg.shape)
        return sunPos

    def transform(self, cloudState, time):
        """ Transform our cloud map according to cloudState
        
        TODO CloudMap is now peering at the guts of cloudState, which makes
        the CloudState class effectively only useful as an argument wrapper.
        Calling cloudState.transform(cloudMap, time) is also dicey though
        because cloudState then needs to know about cloudMap.cloudData. 

        Pixels which are translated from invalid points to valid points take
        the value of the old pixel.

        @returns    the translated map
        @param      cloudState: the CloudState for the transformation
        @param      time: the amount of time to propagate cloudState through
        """
        
        direction = np.round(np.array(cloudState.vel) * time).astype(int)

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
        transformedData = paddedData[cropY[0]:cropY[1],cropX[0]:cropX[1]]

        # replace all pixels which are -1 with the value they used to be
        (invalidY, invalidX) = np.where((transformedData == -1) &
                                        (self.cloudData  != -1))
        transformedData[invalidY,invalidX] = self.cloudData[invalidY,invalidX]

        # np.roll translates with wrap around but we probably don't want this
        #translatedCart = np.roll(cart, direction[0], axis=0)
        #translatedCart = np.roll(translatedCart, direction[1], axis=1)

        #TODO need to deal with mapId better
        mId = self.mapId + str(np.random.random())
        return CloudMap(mId, transformedData, sunPos = self.sunPos + direction)

    def plot(self, maxPixel, title=""):
        plt.figure(title)
        pylab.imshow(self.cloudData, vmax = maxPixel, cmap=plt.cm.jet)
        plt.colorbar()
        # uncomment plt.show() to pause after each call to this function
        #plt.show()

    def max(self):
        return np.max(self.cloudData)

    def std(self):
        return np.std(self.cloudData[self.validMask])

    def mean(self):
        return np.mean(self.cloudData[self.validMask])


def fromHpix(mapId, hpix):
    """ Convert a healpix image to a cartesian cloud map

    @returns    a CloudMap object with the data from the hpix
    @param      mapId: a unique identifier for the map
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
    x = np.repeat([np.arange(-xyCent, xyCent)], xyMax, axis=0).T
    y = np.repeat([np.arange(-xyCent, xyCent)], xyMax, axis=0)
    cart = np.swapaxes([y,x],0,2)
    
    # calculate theta and phi of each pixel in the cartesian map
    r = np.linalg.norm(cart, axis=2) 
    phi = np.arctan2(y, x).T
    theta = np.arctan(r / z)

    # ipixes is an array of pixel indices corresponding to theta and phi
    ipixes = hp.ang2pix(nside, theta, phi)

    # move back from physical coordinates to array indices
    y += xyCent
    x += xyCent
    y = y.astype(int) 
    x = x.astype(int)

    # set the cloud data pixels to the corresponding hpix pixels
    cloudData = np.zeros((xyMax, xyMax))
    cloudData[y.flatten(),x.flatten()] = hpix[ipixes.flatten()]

    return CloudMap(mapId, cloudData)

def toHpix(cloudMap):
    """ Convert a CloudMap to a healpix image

    @returns    a healpix image of the clouds
    @param      cloudMap: a CloudMap with the cloud cover data
    
    For each pixel in hpix, sample from the corresponding pixel in cloudMap
    """

    hpix = np.zeros(npix)
    (theta, phi) = hp.pix2ang(nside, np.arange(npix))

    r = np.tan(theta) * z
    x = np.floor(r * np.cos(phi)).astype(int)
    y = np.floor(r * np.sin(phi)).astype(int)
    
    # ignore all pixels with zenith angle higher than thetaMax
    x = x[theta < thetaMax]
    y = y[theta < thetaMax]
    ipixes = np.arange(npix)[theta < thetaMax]

    hpix[ipixes] = cloudMap[x + xyCent, y + xyCent]

    return hpix

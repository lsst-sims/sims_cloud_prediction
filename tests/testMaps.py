import numpy as np
import healpy as hp
import unittest
import lsst.sims.cloud.prediction as cp
import lsst.utils.tests


class TestMaps(unittest.TestCase):

    def testmaps(self):
        nside = 32
        npix = hp.nside2npix(nside)
        clouds_hp = np.zeros(npix)

        zenith_angle, tmp = hp.pix2ang(nside, np.arange(npix))

        # set a blob at zenith to be cloudy
        hp_to_cloud = np.where(zenith_angle < np.radians(10.))
        clouds_hp[hp_to_cloud] = 1

        cm = cp.fromHpix(0, clouds_hp)
        clouds_hp_back = cp.toHpix(cm)

        diff = np.abs(clouds_hp[hp_to_cloud]-clouds_hp_back[hp_to_cloud]).sum() / float(hp_to_cloud[0].size)
        # demand that 90% of cloudy pixels return as cloudy after round tripping
        assert(np.abs(diff) < 0.1)

        # Unmasked and originally cloudless.
        cloudless = np.where((clouds_hp_back != -1) & (clouds_hp == 0))
        diff = clouds_hp_back[cloudless].sum() / float(np.size(cloudless[0]))
        # demand 90% of the cloudless pixels stay cloudless.
        assert(diff < 0.1)


class TestMemory(lsst.utils.tests.MemoryTestCase):
    pass


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    lsst.utils.tests.init()
unittest.main()

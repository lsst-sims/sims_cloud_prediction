# Cloud Prediction for LSST

## Introduction
Cloud motion is predictable on a 10 minute timescale (and potentially longer with more sophisticated models), and the LSST scheduler has the opportunity to use such predictions when generating its schedules. Without predicted cloud data, the scheduler may plan out an optimal path, only to have it disrupted by an encroaching cloud that could have been avoided more effectively. To solve this problem, we have created this cloud prediction model, which uses streaming cloud data to predict cloud propagation.

We model the sky as 2D cartesian sheet with clouds that all move together at a constant velocity. We calculate the velocity with pairwise cloud map comparisons spaced apart in time, and we use a median filter to reduce noise.

## Code Organization
The `CloudMap` class represents a map of the clouds at a particular time. A series of `CloudMap` instances are managed by a `CloudServer` instance. The `CloudServer` class processes streaming cloud map data that is passed in as it becomes available. It also provides a prediction method that returns a predicted cloud map based on the data currently available in the `CloudServer`. 

The `Cloudmap` objects compute the 2D FFT of the cloud sheet, and velocities are computed by cross-correlating pairs of `Cloudmaps`. 

## Usage

During full operations, an all-sky camera will be periodically imaging the whole sky in order to track clouds. Every time an image is taken, a HEALPix representation of the image will be sent to the cloud prediction code. There, it needs to be converted to a `CloudMap` using the `CloudMap.fromHpix` method. The resulting `CloudMap` needs to be posted to a running `CloudServer` instance using the `CloudServer.postCloudMap` function. Then, whenever the scheduler needs a cloud prediction, it can call `CloudServer.predCloudMap(mjd)`. 

Alternatively, under the publish/subscribe architecture, whatever code is hosting the cloud prediction code can periodically call `CloudServer.predCloudMap` and then publish the result.

## Data

Sample data is available at `lsst-dev01.ncsa.illinois.edu:/datasets/public_html/sim-data/cloudy_night_example`

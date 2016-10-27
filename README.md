# Cloud Prediction for LSST

## Introduction
Cloud motion is predictable on a 10 minute timescale (and potentially longer with more sophisticated models), and the LSST scheduler has the opportunity to use such predictions when generating its schedules. Without predicted cloud data, the scheduler may plan out an optimal path, only to have it disrupted by an encroaching cloud that could have been avoided more effectively. To solve this problem, we have created this cloud prediction model, which uses streaming cloud data to predict cloud propagation.

We modeled the sky as 2D cartesian sheet with clouds that all move together at a constant velocity. We calculate the velocity with pairwise cloud map comparisons spaced apart in time, and we use a median filter to reduce noise.

## Code Organization
The `CloudMap` class represents a map of the clouds at a particular time. A series of `CloudMap` instances are managed by a `CloudServer` instance. The `CloudServer` class processes streaming cloud map data that is passed in as it becomes available. It also provides a prediction method that returns a predicted cloud map based on the data currently available in the `CloudServer`. 

The velocity estimation itself is performed in a subclass of `CloudStateEstimator`. Currently there are two such subclasses: `RmseEstimator` and `FftEstimator`. `RmseEstimator` estimates the displacement between two cloud maps by performing gradient descent on `(deltaX, deltaY)`, where the loss function is the rmse between one map and the other map shifted by `(deltaX, deltaY)`. `FftEstimator` estimates the displacement between two cloud maps by fourier transforming the maps and measuring the phase difference. This method is thanks to Ian Sullivan and is much faster than `RmseEstimator`, but I haven't been able to get it to give as consistent results as `RmseEstimator`.

## Usage

During full operations, an all-sky camera will be periodically imaging the whole sky in order to track clouds. Every time an image is taken, a HEALPix representation of the image will be sent to the cloud prediction code. There, it needs to be converted to a `CloudMap` using the `CloudMap.fromHpix` method. The resulting `CloudMap` needs to be posted to a running `CloudServer` instance using the `CloudServer.postCloudMap` function. Then, whenever the scheduler needs a cloud prediction, it can call `CloudServer.predCloudMap(mjd)`. 

Alternatively, under the publish/subscribe architecture, whatever code is hosting the cloud prediction code can periodically call `CloudServer.predCloudMap` and then publish the result.

Currently, the code is exercised using the `testPredClouds.py` file. This file reads in data from fits files, presents it to a `CloudServer` as if in real time, and then queries the `CloudServer` for predictions. It then calculates the accuracy of the predictions compared to the true clouds at the later time, and it visualizes the results. You can run this file with the command `python testPredClouds.py 111515`. The `111515` specifies which files the script should use (the ones from the date 11/15/15), but because I've only uploaded data from one day, the only valid input is `111515`.

## Data

I have uploaded sample data to `lsst-dev.ncsa.illinois.edu:/home/drothchi`

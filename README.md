# Remote Sensing
This is a repository with scripts that read data from satellites and store bands and vegetation indexes as time series in a MongoDB collection.


![Sentinel-2_L1C-timelapse (3)](https://user-images.githubusercontent.com/55694211/163181232-b16990c0-f197-491c-b6cc-fb03dd7a7c78.gif)

## Objectives
We provide a set of scripts which scan land crops in Egypt over the period 2016-2022. From the satellite's sensor we extract the information needed to construct different vegetation indexes. We store it in a MongoDB collection as time series values. Then, we apply a filter and resampling processing which stores in another collection the new time series data.

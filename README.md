# cosmoNODE

## Repository Overview

Neural Ordinary Differential Equations for Astronomical Sky Surveys

I am using [Kaggle PLAsTiCC-2018](https://www.kaggle.com/c/PLAsTiCC-2018/)

Neural differential equations have not been applied to many fields of ML, including astronomy.

The goal is to see if there is any improvement in the classification accuracy using NODEs.

Given that ODEs are commonly used for time dependent systems,
my hypothesis is that there will be a measurable improvement versus LSTM/RNN classification methods.

## Code

I am working on implementing the PLAsTiCC classification model in Julia, using the DifferentialEquations.jl and Flux.jl packages.

Currently I have a neural stochastic differential equation (NSDE) learn to model the flux of a given object over time. 

## Links

[Project Overview document](https://drive.google.com/open?id=1dDKOfZrUGG_9MTxTWis1rhZ4L-IAFqjq8vEfahfMiVs)

[Overleaf Paper](https://www.overleaf.com/read/pznqtfcgzxyp)

[Colab Notebooks](https://drive.google.com/open?id=1twyeXpB2EeFEyGj7Y61C9KN7vSuHcUv0)

`contact (anandj @ uchicago dot edu)`

## Credit

TODO add a whole bunch of papers
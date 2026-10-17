# Hydroviewer Data Pipeline AGENTS.MD

## Project Overview

This the backend data processing repository for the HydroViewer App, which is a webapp developed for delivering monthly hydro-meteorology forecast from Land Data Assimilation System at the NASA Goddard Space Flight Centre.

The current version of the repository compresses of two parts, `updater.py` and `fire_risk`. The `updater.py` is responsible for the proper functioning of our data service, it runs once a month to update our LIS/LDAS forecast date to the latest release.

The `fire_risk` module is the compute scripts for our data products, which uses the monthly forecast data as input. Right now, we are using them for Canadian Fire Weather Index calculation.

## Dev Environment Requirements

`python -m pip install -f requirement.txt`

## Build and test commands

## Code style guidelines

## Testing instructions

## Security considerations

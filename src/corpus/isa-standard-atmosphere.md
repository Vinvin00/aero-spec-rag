---
title: The International Standard Atmosphere (ISA) Model
topic: atmosphere
source_type: derived
---

# The International Standard Atmosphere (ISA) Model

The International Standard Atmosphere is a piecewise model of how temperature,
pressure, and density vary with geopotential altitude. It is not a forecast of
any particular day's weather; it is an agreed-upon reference so that engineers
comparing vehicle performance are comparing against the same air.

## Structure of the model

The model divides the lower atmosphere into layers. Within each layer the
temperature is assumed to vary linearly with altitude at a fixed lapse rate. The
troposphere, from sea level to about 11 km, has a constant negative lapse rate:
the air cools steadily with height. Above it the lower stratosphere is treated
as isothermal up to roughly 20 km, after which the temperature begins to rise
again.

Given the temperature profile, pressure follows from hydrostatic equilibrium
combined with the ideal gas law. Density is then recovered from the equation of
state rather than being modelled independently.

## Sea-level reference conditions

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| air density | 1.225 | kg/m^3 | ISA sea level, standard day |
| pressure | 101325 | Pa | ISA sea level |
| temperature | 288.15 | K | ISA sea level, equals 15 degrees Celsius |
| speed of sound | 340.3 | m/s | ISA sea level, dry air |
| lapse rate | -0.0065 | K/m | troposphere, 0 to 11 km |
| gravity | 9.80665 | m/s^2 | standard acceleration, used throughout ISA |
| specific gas constant | 287.05 | J/(kg*K) | dry air |
| ratio of specific heats | 1.4 | dimensionless | dry air, gamma |

## Why guidance simulations care

A trajectory model that integrates drag needs density at every timestep, and
density falls by roughly a factor of ten between sea level and 16 km. Using a
sea-level value throughout a high-altitude engagement overestimates drag
severely, which in turn understates range. Most simulation codebases therefore
carry an ISA routine, or an exponential fit to it, as a core utility.

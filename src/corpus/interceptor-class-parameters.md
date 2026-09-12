---
title: Illustrative Interceptor Class Parameter Ranges
topic: vehicle_parameters
source_type: illustrative
---

# Illustrative Interceptor Class Parameter Ranges

**These numbers are invented order-of-magnitude placeholders.** They exist so
that a simulation has physically coherent inputs to run with and so that a
retrieval pipeline has something to ground against. They are not the
specifications of any real weapon system, and nothing here is derived from
restricted sources. Treat every figure as a rounded teaching value.

## Generic short-range agile interceptor

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| launch mass | 90 | kg | illustrative short-range class |
| body diameter | 0.16 | m | illustrative |
| body length | 2.9 | m | illustrative |
| reference area | 0.0201 | m^2 | frontal area from the diameter above |
| drag coefficient | 0.25 | dimensionless | illustrative subsonic axial value |
| burn time | 4 | s | illustrative single-pulse motor |
| peak Mach | 2.5 | dimensionless | illustrative |
| maximum lateral acceleration | 30 | g | illustrative airframe limit |
| navigation gain | 4 | dimensionless | illustrative baseline tuning |

## Generic medium-range interceptor

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| launch mass | 700 | kg | illustrative medium-range class |
| body diameter | 0.34 | m | illustrative |
| body length | 5.5 | m | illustrative |
| reference area | 0.0908 | m^2 | frontal area from the diameter above |
| drag coefficient | 0.2 | dimensionless | illustrative supersonic cruise value |
| burn time | 12 | s | illustrative boost-sustain motor |
| peak Mach | 4 | dimensionless | illustrative |
| maximum lateral acceleration | 20 | g | illustrative airframe limit |
| intercept altitude | 20000 | m | illustrative upper engagement bound |

## How to use these

Pick one class, hold it fixed, and vary one parameter at a time. The value of a
set like this in a portfolio simulation is that the trends it produces are
believable, not that the absolute numbers are right.

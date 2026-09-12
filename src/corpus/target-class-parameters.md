---
title: Illustrative Target Class Parameter Ranges
topic: vehicle_parameters
source_type: illustrative
---

# Illustrative Target Class Parameter Ranges

**As with the interceptor figures, everything here is an invented placeholder**
chosen to be physically self-consistent and roughly the right order of
magnitude. No figure describes a real vehicle.

## Slow non-manoeuvring target

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| target speed | 200 | m/s | illustrative subsonic cruise |
| target altitude | 1000 | m | illustrative low-level ingress |
| target lateral acceleration | 2 | g | illustrative gentle turn limit |
| reference area | 0.15 | m^2 | illustrative frontal area |
| drag coefficient | 0.3 | dimensionless | illustrative |

## Fast manoeuvring target

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| target speed | 600 | m/s | illustrative supersonic dash |
| target altitude | 12000 | m | illustrative high-altitude ingress |
| target lateral acceleration | 9 | g | illustrative hard evasive turn |
| reference area | 0.05 | m^2 | illustrative frontal area |
| drag coefficient | 0.22 | dimensionless | illustrative |

## Why target assumptions dominate results

Miss distance in a proportional navigation study is far more sensitive to the
assumed target manoeuvre than to most interceptor parameters. A target that
pulls a hard turn inside the last second of flight forces a commanded
acceleration that is a multiple of the steady-state requirement, and that is
usually where the airframe limit binds. Any reported miss distance should
therefore be quoted alongside the target manoeuvre model it assumed, or it means
very little.

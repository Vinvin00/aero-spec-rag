---
title: ISA Density and Pressure Against Altitude
topic: atmosphere
source_type: derived
---

# ISA Density and Pressure Against Altitude

The values below are computed from the standard ISA layer equations for a
standard day. They are tabulated here as a convenience for order-of-magnitude
checks; a simulation should evaluate the layer equations directly rather than
interpolating a coarse table.

## Tabulated profile

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| air density | 1.225 | kg/m^3 | ISA at 0 m altitude |
| air density | 1.112 | kg/m^3 | ISA at 1000 m altitude |
| air density | 1.007 | kg/m^3 | ISA at 2000 m altitude |
| air density | 0.9093 | kg/m^3 | ISA at 3000 m altitude |
| air density | 0.7364 | kg/m^3 | ISA at 5000 m altitude |
| air density | 0.5258 | kg/m^3 | ISA at 8000 m altitude |
| air density | 0.4135 | kg/m^3 | ISA at 10000 m altitude |
| air density | 0.3639 | kg/m^3 | ISA at 11000 m altitude |
| air density | 0.1948 | kg/m^3 | ISA at 15000 m altitude |
| air density | 0.0889 | kg/m^3 | ISA at 20000 m altitude |
| air density | 0.0184 | kg/m^3 | ISA at 30000 m altitude |
| pressure | 22632 | Pa | ISA at 11000 m altitude |
| pressure | 5474 | Pa | ISA at 20000 m altitude |
| temperature | 216.65 | K | ISA tropopause, 11 to 20 km |
| scale height | 8500 | m | exponential fit, lower atmosphere |

## The exponential approximation

For quick work the lower atmosphere is often approximated as an exponential:
density equals the sea-level value multiplied by the exponential of minus
altitude divided by a scale height. A scale height near 8500 m reproduces the
true ISA profile to within a few percent through the troposphere, and degrades
above the tropopause where the isothermal layer changes the slope.

This approximation is attractive in a guidance loop because it is cheap, smooth,
and analytically differentiable, which matters if the trajectory optimiser needs
density gradients.

---
title: The Drag Equation and Its Terms
topic: aerodynamics
source_type: derived
---

# The Drag Equation and Its Terms

Aerodynamic drag on a body moving through air is modelled as one half times the
air density, times the square of the speed relative to the air mass, times a
reference area, times a dimensionless drag coefficient.

Each term deserves a note:

**Density** comes from the atmosphere model and varies with altitude, so drag
falls off sharply as a vehicle climbs even at constant airspeed.

**Velocity squared** is the dominant term in most engagements. Doubling speed
quadruples drag force, which is why boost-phase deceleration after burnout is so
pronounced.

**Reference area** is a bookkeeping choice, not a physical property. For a
slender body the convention is usually the maximum cross-sectional area of the
body; for a wing it is the planform area. A drag coefficient is meaningless
without knowing which reference area it was normalised against, and mixing the
two conventions is a common source of factor-of-several errors.

**Drag coefficient** absorbs everything the simple formula does not capture:
shape, surface finish, angle of attack, and the Reynolds and Mach dependence of
the flow.

## Dependence on Mach number

The drag coefficient of a slender body is roughly constant in the subsonic
regime, rises steeply through the transonic range as shock waves form, peaks a
little above Mach 1, and then falls slowly through the supersonic regime without
returning to its subsonic value. A simulation that treats the coefficient as a
single constant across a boost from rest to Mach 4 will misplace the peak drag
by a wide margin.

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| drag coefficient | 0.1 to 1.5 | dimensionless | plausible envelope for streamlined to bluff bodies |
| transonic peak Mach | 1.1 | dimensionless | approximate location of peak wave drag |

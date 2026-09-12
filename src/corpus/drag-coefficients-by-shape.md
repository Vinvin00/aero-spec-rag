---
title: Typical Drag Coefficient Ranges by Body Shape
topic: aerodynamics
source_type: illustrative
---

# Typical Drag Coefficient Ranges by Body Shape

The figures below are round, illustrative values for incompressible or low
subsonic flow, normalised against frontal area unless noted. They are intended
for sanity checks and first-pass sizing, not for design work. Real coefficients
depend on Reynolds number, surface roughness, angle of attack, and Mach number,
and a measured or computed value should always displace a table lookup.

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| drag coefficient | 1.1 | dimensionless | flat plate normal to flow, bluff body |
| drag coefficient | 1.05 | dimensionless | cube, face-on |
| drag coefficient | 0.47 | dimensionless | sphere, subcritical Reynolds number |
| drag coefficient | 0.2 | dimensionless | sphere, supercritical Reynolds number after drag crisis |
| drag coefficient | 0.82 | dimensionless | long cylinder, flow normal to axis |
| drag coefficient | 0.3 | dimensionless | blunt-nosed slender body, subsonic |
| drag coefficient | 0.15 | dimensionless | slender finned body, subsonic, axial flow |
| drag coefficient | 0.04 | dimensionless | streamlined teardrop half-body, best case |

## The drag crisis

A smooth sphere shows a striking discontinuity: as Reynolds number crosses
roughly half a million the boundary layer transitions to turbulence, the
separation point moves aft, the wake narrows, and the coefficient drops by more
than half. This is the reason a golf ball is dimpled. Any table that quotes a
single value for a sphere is implicitly picking a side of that transition.

## Slender bodies at angle of attack

For a finned body the axial coefficient understates the total force as soon as
the body flies at incidence. Induced drag grows roughly with the square of the
angle of attack, so a hard manoeuvre both bleeds energy and shortens range. A
guidance simulation that commands large lateral accelerations without modelling
induced drag will predict unrealistically flat energy loss.

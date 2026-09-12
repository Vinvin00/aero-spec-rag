---
title: Engagement Geometry and Closing Velocity
topic: guidance
source_type: derived
---

# Engagement Geometry and Closing Velocity

## Line of sight, range rate, and time to go

The engagement is described by the relative position and relative velocity of
the two vehicles. The line of sight is the unit vector from pursuer to target.
Range rate is the component of relative velocity along that line; closing
velocity is its negative, positive when the range is shrinking. Time to go, in
the simplest estimate, is range divided by closing velocity, which is exact only
for constant closing velocity and is therefore biased in manoeuvring or
decelerating engagements.

The line-of-sight rotation rate is the component of relative velocity
perpendicular to the line of sight, divided by the range. As range collapses at
the end of the engagement this quantity becomes numerically delicate: a small
lateral error divided by a small range produces a large and noisy rate. Most
implementations either freeze the guidance command or hand over to a fuzing
logic inside some small terminal radius rather than letting the divide blow up.

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| closing velocity | 300 to 1800 | m/s | illustrative range across head-on and tail-chase geometries |
| terminal guidance cutoff range | 10 | m | illustrative radius inside which commands are frozen |
| simulation timestep | 0.001 to 0.01 | s | typical fixed-step integration for terminal homing |

## Head-on versus tail chase

A head-on geometry has high closing velocity, short time to go, and demanding
acceleration requirements but little time for the target to react. A tail chase
has low closing velocity, long flight time, and correspondingly large energy
loss to drag. The same guidance gain behaves very differently in the two cases,
which is why gain tuning is validated across a geometry matrix rather than at a
single nominal engagement.

## Integration fidelity

Terminal homing resolves in a few seconds and the interesting dynamics live in
the last fraction of a second. A fixed-step integrator at a millisecond is
usually adequate; at ten milliseconds the reported miss distance starts to
depend visibly on the step size, which is a reliable sign the result is a
numerical artefact rather than physics.

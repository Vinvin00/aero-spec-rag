---
title: Selecting the Proportional Navigation Gain
topic: guidance
source_type: illustrative
---

# Selecting the Proportional Navigation Gain

The navigation gain, conventionally written as N prime, is the single most
consequential tuning parameter in a proportional navigation loop. The values
below are the ranges commonly used in open-literature simulation studies and in
teaching examples; they are illustrative starting points, not the tuning of any
fielded system.

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| navigation gain | 3 to 5 | dimensionless | usual working range for terminal homing |
| navigation gain | 4 | dimensionless | common default for a first simulation |
| navigation gain | 2 to 6 | dimensionless | full plausible envelope including edge cases |
| navigation gain | 3 | dimensionless | minimum for acceptable convergence against a steady target |

## Why the range is narrow

Below a gain of about 3 the line-of-sight rate is not driven down fast enough
and the required acceleration piles up at the end of the engagement, exactly
where the pursuer has the least energy and control authority to deliver it.
Above about 5 the loop begins to amplify seeker noise and the airframe is asked
for acceleration it cannot supply; the command saturates and the effective gain
collapses anyway. The optimal value under idealised assumptions with no lag
falls at 3, and practical implementations sit slightly higher to buy margin
against target manoeuvre and estimation lag.

## Interaction with lag

Every real loop has lag: seeker filtering, autopilot response, and actuator
dynamics. The usual rule of thumb is that higher gains are needed as the ratio
of total system lag to time-to-go grows, but the benefit is bounded because lag
also erodes stability margin. Tuning is therefore done against a specific
airframe and seeker, in simulation, over a matrix of engagement geometries
rather than at a single nominal condition.

| Quantity | Value | Unit | Notes |
| --- | --- | --- | --- |
| guidance update rate | 50 to 200 | Hz | typical digital loop rate in simulation studies |

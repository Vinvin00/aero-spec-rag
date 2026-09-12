---
title: Proportional Navigation Fundamentals
topic: guidance
source_type: derived
---

# Proportional Navigation Fundamentals

Proportional navigation is the workhorse terminal guidance law for intercepting
a moving target. Its premise is geometric: if the line of sight from pursuer to
target holds a constant bearing while the range closes, the two are on a
collision course. Guidance therefore does not try to point the pursuer at the
target; it tries to drive the rotation rate of the line of sight to zero.

## The law

The commanded lateral acceleration is the navigation gain times the closing
velocity times the line-of-sight rotation rate. In the true form the
acceleration is commanded perpendicular to the line of sight; in the pure form
it is commanded perpendicular to the pursuer's velocity vector. Augmented
proportional navigation adds a term proportional to the estimated target
acceleration, which materially improves performance against a manoeuvring
target at the cost of needing that estimate.

## Behaviour

Because the command scales with closing velocity, the law naturally becomes more
aggressive in fast head-on geometries and gentler in tail chases. Against a
non-manoeuvring target and with no lag, the required lateral acceleration
decays toward zero as intercept approaches, which is the signature of a
well-tuned engagement. Rising commanded acceleration in the last second usually
indicates either a target manoeuvre, a seeker noise problem, or a gain that is
too low for the geometry.

## Practical limitations

The law assumes the pursuer can deliver the acceleration it commands. Once the
command saturates against the airframe limit, or against available fin
authority at low dynamic pressure, performance degrades and miss distance
grows. Seeker noise is amplified by the gain, so a high gain trades manoeuvre
margin for noise sensitivity, and most implementations low-pass filter the
line-of-sight rate estimate before using it.

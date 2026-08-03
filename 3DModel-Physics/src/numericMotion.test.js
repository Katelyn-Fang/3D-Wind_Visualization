import test from "node:test";
import assert from "node:assert/strict";
import {
  createNumericMotionPlan,
  INCH_TO_METERS,
  sampleNumericMotion,
} from "./numericMotion.js";
import { inferWindFromMotion } from "./windModel.js";

function plan(overrides = {}) {
  return createNumericMotionPlan({
    startPosition: { x: 1, y: 2, z: 3 },
    offsetInches: { x: 10, y: -5, z: 20 },
    startAttitude: { roll: 2, pitch: 3, yaw: 4 },
    attitudeOffsetDegrees: { roll: 8, pitch: -3, yaw: 26 },
    durationSeconds: 2,
    ...overrides,
  });
}

test("converts inch offsets and reaches the exact target pose", () => {
  const motion = plan();
  const end = sampleNumericMotion(motion, 2);

  assert.equal(motion.offsetMeters.x, 10 * INCH_TO_METERS);
  assert.deepEqual(end.position, motion.targetPosition);
  assert.deepEqual(end.attitude, motion.targetAttitude);
  assert.equal(end.done, true);
});

test("starts and ends at rest", () => {
  const motion = plan();
  const start = sampleNumericMotion(motion, 0);
  const end = sampleNumericMotion(motion, motion.durationSeconds);

  for (const sample of [start, end]) {
    assert.ok(Math.hypot(...Object.values(sample.velocity)) < 1e-12);
    assert.ok(Math.hypot(...Object.values(sample.acceleration)) < 1e-12);
  }
});

test("shorter duration produces higher speed and acceleration", () => {
  const slow = plan({ durationSeconds: 4 });
  const fast = plan({ durationSeconds: 2 });
  const slowSample = sampleNumericMotion(slow, 1);
  const fastSample = sampleNumericMotion(fast, 0.5);

  const slowSpeed = Math.hypot(...Object.values(slowSample.velocity));
  const fastSpeed = Math.hypot(...Object.values(fastSample.velocity));
  const slowAcceleration = Math.hypot(
    ...Object.values(slowSample.acceleration),
  );
  const fastAcceleration = Math.hypot(
    ...Object.values(fastSample.acceleration),
  );

  assert.ok(Math.abs(fastSpeed / slowSpeed - 2) < 1e-10);
  assert.ok(Math.abs(fastAcceleration / slowAcceleration - 4) < 1e-10);
});

test("distance and duration change the inferred wind strength", () => {
  const windAtQuarter = (motion) => {
    const sample = sampleNumericMotion(
      motion,
      motion.durationSeconds * 0.25,
    );
    return inferWindFromMotion({
      yawDegrees: 0,
      pitchDegrees: 0,
      rollDegrees: 0,
      velocityX: sample.velocity.x,
      velocityY: sample.velocity.y,
      velocityZ: sample.velocity.z,
      accelerationX: sample.acceleration.x,
      accelerationY: sample.acceleration.y,
      accelerationZ: sample.acceleration.z,
    }).speed;
  };

  const baseline = plan({
    offsetInches: { x: 24, y: 0, z: 0 },
    durationSeconds: 2,
  });
  const farther = plan({
    offsetInches: { x: 48, y: 0, z: 0 },
    durationSeconds: 2,
  });
  const faster = plan({
    offsetInches: { x: 24, y: 0, z: 0 },
    durationSeconds: 1,
  });

  assert.ok(windAtQuarter(farther) > windAtQuarter(baseline));
  assert.ok(windAtQuarter(faster) > windAtQuarter(baseline));
});

test("axis-only offsets infer wind along the requested axis", () => {
  for (const axis of ["x", "y", "z"]) {
    for (const sign of [-1, 1]) {
      const offsetInches = { x: 0, y: 0, z: 0 };
      offsetInches[axis] = 50 * sign;
      const motion = plan({ offsetInches });
      const sample = sampleNumericMotion(
        motion,
        motion.durationSeconds * 0.25,
      );
      const wind = inferWindFromMotion({
        yawDegrees: 0,
        pitchDegrees: 0,
        rollDegrees: 0,
        velocityX: sample.velocity.x,
        velocityY: sample.velocity.y,
        velocityZ: sample.velocity.z,
        accelerationX: sample.acceleration.x,
        accelerationY: sample.acceleration.y,
        accelerationZ: sample.acceleration.z,
      });

      assert.ok(wind[axis] * sign > 0);
      for (const otherAxis of ["x", "y", "z"]) {
        if (otherAxis !== axis) {
          assert.ok(Math.abs(wind[otherAxis]) < 1e-12);
        }
      }
    }
  }
});

test("rejects invalid durations", () => {
  assert.throws(
    () => plan({ durationSeconds: 0 }),
    /between 0.1 and 60 seconds/,
  );
});

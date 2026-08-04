import test from "node:test";
import assert from "node:assert/strict";
import {
  createNumericMotionPlan,
  INCH_TO_METERS,
  sampleNumericMotion,
} from "./numericMotion.js";
import {
  createMlTelemetrySample,
  MODEL_COORDINATE_ORIGIN,
  sceneToModelCoordinates,
} from "./mlTelemetry.js";
import {
  createTrajectorySampleTimes,
  prepareMlTrajectory,
  sampleMlTrajectory,
} from "./mlTrajectory.js";

function plan(overrides = {}) {
  return createNumericMotionPlan({
    startPosition: { x: 0, y: 1.35, z: 0 },
    offsetInches: { x: 24, y: 0, z: 12 },
    startAttitude: { roll: 0, pitch: 0, yaw: 0 },
    attitudeOffsetDegrees: { roll: 5, pitch: -3, yaw: 30 },
    durationSeconds: 2,
    ...overrides,
  });
}

test("converts inch offsets and reaches the exact target pose", () => {
  const motion = plan();
  const end = sampleNumericMotion(motion, motion.durationSeconds);

  assert.equal(motion.offsetMeters.x, 24 * INCH_TO_METERS);
  assert.equal(motion.offsetMeters.z, 12 * INCH_TO_METERS);
  assert.deepEqual(end.position, motion.targetPosition);
  assert.deepEqual(end.attitude, motion.targetAttitude);
  assert.equal(end.done, true);
});

test("minimum-jerk motion starts and ends at rest", () => {
  const motion = plan();
  for (const elapsed of [0, motion.durationSeconds]) {
    const sample = sampleNumericMotion(motion, elapsed);
    assert.ok(Math.hypot(...Object.values(sample.velocity)) < 1e-12);
    assert.ok(Math.hypot(...Object.values(sample.acceleration)) < 1e-12);
  }
});

test("shorter duration produces faster ML telemetry changes", () => {
  const slow = plan({ durationSeconds: 4 });
  const fast = plan({ durationSeconds: 2 });
  const slowSample = sampleNumericMotion(slow, 1);
  const fastSample = sampleNumericMotion(fast, 0.5);

  assert.ok(
    Math.abs(
      Math.hypot(...Object.values(fastSample.velocity)) /
        Math.hypot(...Object.values(slowSample.velocity)) -
        2,
    ) < 1e-10,
  );
});

test("maps Three.js east, up, and north axes into model coordinates", () => {
  const coordinates = sceneToModelCoordinates({ x: 10, y: 2, z: 20 });

  assert.ok(coordinates.x > MODEL_COORDINATE_ORIGIN.longitude);
  assert.ok(coordinates.y > MODEL_COORDINATE_ORIGIN.latitude);
  assert.equal(coordinates.z, MODEL_COORDINATE_ORIGIN.altitudeM + 2);
});

test("numeric samples become Extra Trees API telemetry", () => {
  const motion = plan();
  const halfway = sampleNumericMotion(motion, 1);
  const payload = createMlTelemetrySample({
    sessionId: "numeric-test",
    elapsedSeconds: halfway.elapsedSeconds,
    position: halfway.position,
    attitudeDegrees: halfway.attitude,
    batteryVoltage: 15.4,
    batteryCurrent: 5.2,
  });

  assert.equal(payload.session_id, "numeric-test");
  assert.equal(payload.elapsed_s, 1);
  assert.ok(payload.x > MODEL_COORDINATE_ORIGIN.longitude);
  assert.ok(payload.y > MODEL_COORDINATE_ORIGIN.latitude);
  assert.equal(payload.roll, halfway.attitude.roll * Math.PI / 180);
  assert.equal(payload.pitch, halfway.attitude.pitch * Math.PI / 180);
  assert.equal(payload.yaw, halfway.attitude.yaw * Math.PI / 180);
  assert.equal(payload.battery_v, 15.4);
  assert.equal(payload.battery_c, 5.2);
});

test("rejects invalid durations", () => {
  assert.throws(
    () => plan({ durationSeconds: 0 }),
    /between 0.1 and 60 seconds/,
  );
});

test("ML trajectory sampling includes both endpoints and caps long runs", () => {
  assert.deepEqual(createTrajectorySampleTimes(0.2), [0, 0.1, 0.2]);
  const longRun = createTrajectorySampleTimes(60);
  assert.equal(longRun.length, 121);
  assert.equal(longRun[0], 0);
  assert.equal(longRun.at(-1), 60);
});

test("prepares batched Extra Trees telemetry for the complete path", async () => {
  const payloads = [];
  const predictions = await prepareMlTrajectory({
    plan: plan({ durationSeconds: 0.2 }),
    sessionId: "trajectory-test",
    batteryVoltage: 15,
    batteryCurrent: 4,
    fetchPredictions: async (batch) => {
      payloads.push(...batch);
      return batch.map((payload) => ({
          u: payload.elapsed_s,
          v: payload.elapsed_s * 2,
          w: 0,
          speed: payload.elapsed_s * Math.sqrt(5),
          direction_confidence: 0.8,
        }));
    },
  });

  assert.deepEqual(payloads.map((payload) => payload.elapsed_s), [0, 0.1, 0.2]);
  assert.equal(predictions.length, 3);
  assert.equal(payloads.at(-1).session_id, "trajectory-test");
});

test("interpolates cached ML vectors at the current animation time", () => {
  const prediction = sampleMlTrajectory([
    {
      elapsedSeconds: 0,
      prediction: { u: 0, v: 2, w: 0, direction_confidence: 0.4 },
    },
    {
      elapsedSeconds: 2,
      prediction: { u: 4, v: 0, w: 2, direction_confidence: 0.8 },
    },
  ], 1);

  assert.equal(prediction.u, 2);
  assert.equal(prediction.v, 1);
  assert.equal(prediction.w, 1);
  assert.equal(prediction.speed, Math.sqrt(6));
  assert.ok(Math.abs(prediction.direction_confidence - 0.6) < 1e-12);
});

import test from "node:test";
import assert from "node:assert/strict";
import * as THREE from "three";
import {
  inferWindFromMotion,
  PHYSICS,
  predictWind,
} from "./windModel.js";
import { inferIdentifiedWind } from "./identifiedBaseline.js";

function telemetry(overrides = {}) {
  return {
    yawDegrees: 0,
    pitchDegrees: 0,
    rollDegrees: 0,
    velocityX: 0,
    velocityY: 0,
    velocityZ: 0,
    accelerationX: 0,
    accelerationY: 0,
    accelerationZ: 0,
    ...overrides,
  };
}

test("hovering level requires no aerodynamic wind force", () => {
  const result = inferWindFromMotion(telemetry());

  assert.ok(result.aerodynamicForce.magnitude < 1e-9);
  assert.ok(result.speed < 1e-9);
});

test("positive east acceleration infers wind toward the east", () => {
  const result = inferWindFromMotion(
    telemetry({ accelerationX: 2 }),
  );

  assert.ok(result.x > 0);
  assert.ok(Math.abs(result.z) < 1e-9);
  assert.ok(result.aerodynamicForce.x > 0);
});

test("a stationary pitched drone requires wind opposing tilted thrust", () => {
  const result = inferWindFromMotion(
    telemetry({ pitchDegrees: 15 }),
  );

  // Positive pitch rotates the thrust vector toward negative x, so the
  // aerodynamic force required to hold position must point toward positive x.
  assert.ok(result.x > 0);
  assert.ok(result.aerodynamicForce.x > 0);
});

test("inferred wind is capped for unstable pointer acceleration", () => {
  const result = inferWindFromMotion(
    telemetry({ accelerationX: 10000 }),
  );

  assert.ok(result.speed <= PHYSICS.maximumWindSpeedMps + 1e-9);
});

test("identified baseline returns a finite horizontal vector", () => {
  const result = inferIdentifiedWind(telemetry());
  assert.ok(Number.isFinite(result.x));
  assert.ok(Number.isFinite(result.z));
  assert.equal(result.y, 0);
  assert.ok(result.speed >= 0);
});

test("yaw rotates identified wind direction without changing its speed", () => {
  const north = inferIdentifiedWind(telemetry({ yawDegrees: 0 }));
  const east = inferIdentifiedWind(telemetry({ yawDegrees: 90 }));
  assert.ok(Math.abs(north.speed - east.speed) < 1e-9);
  assert.ok(Math.abs(north.x - east.z) < 1e-9);
  assert.ok(Math.abs(north.z + east.x) < 1e-9);
});

test("steady numeric field does not oscillate laterally over time", () => {
  const dronePosition = new THREE.Vector3(0, 1.35, 0);
  const samplePosition = new THREE.Vector3(3, 1.35, 1);
  const windTelemetry = {
    estimatedWindX: 5,
    estimatedWindY: 0,
    estimatedWindZ: 0,
  };

  const first = predictWind(
    samplePosition,
    dronePosition,
    0,
    { steadyDirection: true },
    windTelemetry,
  ).clone();
  const later = predictWind(
    samplePosition,
    dronePosition,
    1,
    { steadyDirection: true },
    windTelemetry,
  ).clone();

  assert.ok(first.distanceTo(later) < 1e-12);
});

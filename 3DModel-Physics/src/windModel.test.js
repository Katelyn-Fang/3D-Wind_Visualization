import test from "node:test";
import assert from "node:assert/strict";
import { inferWindFromMotion, PHYSICS } from "./windModel.js";

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

import * as THREE from "three";

/**
 * Hypothetical drone and atmosphere values used by the inverse model.
 * These match 3DModel-Physics so the same numeric input produces the same
 * inferred wind in both applications.
 */
export const INVERSE_PHYSICS = Object.freeze({
  massKg: 1.6,
  gravityMps2: 9.81,
  airDensityKgM3: 1.225,
  dragCoefficient: 1.05,
  referenceAreaM2: 0.18,
  maximumWindSpeedMps: 25,
  minimumForceN: 0.02,
});

const attitudeEuler = new THREE.Euler(0, 0, 0, "YZX");
const attitudeQuaternion = new THREE.Quaternion();
const thrustDirection = new THREE.Vector3();
const gravityForce = new THREE.Vector3(
  0,
  -INVERSE_PHYSICS.massKg * INVERSE_PHYSICS.gravityMps2,
  0,
);
const netForce = new THREE.Vector3();
const thrustForce = new THREE.Vector3();
const requiredAerodynamicForce = new THREE.Vector3();
const velocity = new THREE.Vector3();
const inferredWind = new THREE.Vector3();
const relativeAirflow = new THREE.Vector3();

function finite(value) {
  return Number.isFinite(value) ? value : 0;
}

function updateThrustDirection(telemetry) {
  attitudeEuler.set(
    THREE.MathUtils.degToRad(finite(telemetry.rollDegrees)),
    -THREE.MathUtils.degToRad(finite(telemetry.yawDegrees)),
    THREE.MathUtils.degToRad(finite(telemetry.pitchDegrees)),
    "YZX",
  );
  attitudeQuaternion.setFromEuler(attitudeEuler);
  return thrustDirection
    .set(0, 1, 0)
    .applyQuaternion(attitudeQuaternion)
    .normalize();
}

function calculateAssumedThrust(telemetry) {
  const direction = updateThrustDirection(telemetry);
  const verticalShare = Math.max(direction.y, 0.35);
  const hoverThrust =
    INVERSE_PHYSICS.massKg * INVERSE_PHYSICS.gravityMps2;
  const magnitude = Math.min(
    hoverThrust / verticalShare,
    hoverThrust * 2.3,
  );
  return thrustForce.copy(direction).multiplyScalar(magnitude);
}

export function calculateRequiredAerodynamicForce(telemetry = {}) {
  netForce.set(
    INVERSE_PHYSICS.massKg * finite(telemetry.accelerationX),
    INVERSE_PHYSICS.massKg * finite(telemetry.accelerationY),
    INVERSE_PHYSICS.massKg * finite(telemetry.accelerationZ),
  );
  const thrust = calculateAssumedThrust(telemetry);
  requiredAerodynamicForce
    .copy(netForce)
    .sub(thrust)
    .sub(gravityForce);
  return {
    x: requiredAerodynamicForce.x,
    y: requiredAerodynamicForce.y,
    z: requiredAerodynamicForce.z,
    magnitude: requiredAerodynamicForce.length(),
    thrustMagnitude: thrust.length(),
  };
}

/**
 * Invert quadratic drag to find one hypothetical airflow capable of producing
 * the current velocity, acceleration, and attitude demand.
 */
export function inferWindFromMotion(telemetry = {}) {
  const aerodynamicForce = calculateRequiredAerodynamicForce(telemetry);
  velocity.set(
    finite(telemetry.velocityX),
    finite(telemetry.velocityY),
    finite(telemetry.velocityZ),
  );
  inferredWind.copy(velocity);
  relativeAirflow.set(0, 0, 0);

  if (aerodynamicForce.magnitude >= INVERSE_PHYSICS.minimumForceN) {
    const dragArea =
      INVERSE_PHYSICS.dragCoefficient * INVERSE_PHYSICS.referenceAreaM2;
    const relativeSpeed = Math.sqrt(
      (2 * aerodynamicForce.magnitude) /
        (INVERSE_PHYSICS.airDensityKgM3 * dragArea),
    );
    relativeAirflow
      .set(aerodynamicForce.x, aerodynamicForce.y, aerodynamicForce.z)
      .normalize()
      .multiplyScalar(relativeSpeed);
    inferredWind.add(relativeAirflow);
  }

  if (inferredWind.length() > INVERSE_PHYSICS.maximumWindSpeedMps) {
    inferredWind.setLength(INVERSE_PHYSICS.maximumWindSpeedMps);
  }

  return {
    x: inferredWind.x,
    y: inferredWind.y,
    z: inferredWind.z,
    speed: inferredWind.length(),
    relativeAirSpeed: relativeAirflow.length(),
    aerodynamicForce,
  };
}

const baseFlow = new THREE.Vector3();
const flowDirection = new THREE.Vector3();
const relativePosition = new THREE.Vector3();
const radialDirection = new THREE.Vector3();
const modeledFlow = new THREE.Vector3();
const lateralDirection = new THREE.Vector3();
const centerlineOffset = new THREE.Vector3();

/**
 * Apply the same near-body deflection and downstream wake visualization used
 * by 3DModel-Physics to an inverse-physics wind estimate.
 */
export function predictInversePhysicsWindField(
  samplePosition,
  dronePosition,
  timeSeconds,
  params = {},
  telemetry = {},
) {
  baseFlow.set(
    finite(telemetry.estimatedWindX),
    finite(telemetry.estimatedWindY),
    finite(telemetry.estimatedWindZ),
  );
  const speed = baseFlow.length();
  if (speed < 0.02) return modeledFlow.set(0, 0, 0);

  modeledFlow.copy(baseFlow);
  flowDirection.copy(baseFlow).normalize();
  relativePosition.copy(samplePosition).sub(dronePosition);
  const distance = relativePosition.length();
  const bodyRadius = 1.05;

  if (distance > bodyRadius) {
    radialDirection.copy(relativePosition).divideScalar(distance);
    const radiusRatioCubed = (bodyRadius ** 3) / (distance ** 3);
    const radialFlow = baseFlow.dot(radialDirection);
    modeledFlow
      .copy(baseFlow)
      .multiplyScalar(1 + 0.5 * radiusRatioCubed)
      .addScaledVector(
        radialDirection,
        -1.5 * radiusRatioCubed * radialFlow,
      );
  }

  const downstreamDistance = relativePosition.dot(flowDirection);
  if (downstreamDistance > 0) {
    centerlineOffset
      .copy(relativePosition)
      .addScaledVector(flowDirection, -downstreamDistance);
    const radialDistance = centerlineOffset.length();
    const wakeWidth = 0.75 + downstreamDistance * 0.24;
    const radialFalloff = Math.exp(
      -(radialDistance ** 2) / (2 * wakeWidth ** 2),
    );
    const axialFalloff = Math.exp(-downstreamDistance / 7);
    const deficit = 0.55 * radialFalloff * axialFalloff;
    modeledFlow.addScaledVector(flowDirection, -speed * deficit);

    lateralDirection.set(-flowDirection.z, 0, flowDirection.x);
    if (!params.steadyDirection && lateralDirection.lengthSq() > 1e-8) {
      lateralDirection.normalize();
      const wobble =
        Math.sin(timeSeconds * 3.2 + downstreamDistance * 1.7) *
        speed *
        deficit *
        0.12;
      modeledFlow.addScaledVector(lateralDirection, wobble);
    }
  }
  return modeledFlow;
}

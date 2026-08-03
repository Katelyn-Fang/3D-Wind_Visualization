import * as THREE from "three";

/**
 * Hypothetical drone and atmosphere values used by the inverse model.
 * These values are deliberately explicit: the visualization is an
 * explanatory simulation, not a calibrated flight-dynamics estimator.
 */
export const PHYSICS = Object.freeze({
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
const gravityForce = new THREE.Vector3(0, -PHYSICS.massKg * PHYSICS.gravityMps2, 0);
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

/**
 * Assume the flight controller supplies enough thrust to hold altitude.
 * Tilting the drone therefore introduces a horizontal thrust component.
 */
function calculateAssumedThrust(telemetry) {
  const direction = updateThrustDirection(telemetry);
  const verticalShare = Math.max(direction.y, 0.35);
  const hoverThrust = PHYSICS.massKg * PHYSICS.gravityMps2;
  const magnitude = Math.min(hoverThrust / verticalShare, hoverThrust * 2.3);

  return thrustForce.copy(direction).multiplyScalar(magnitude);
}

export function calculateRequiredAerodynamicForce(telemetry = {}) {
  netForce.set(
    PHYSICS.massKg * finite(telemetry.accelerationX),
    PHYSICS.massKg * finite(telemetry.accelerationY),
    PHYSICS.massKg * finite(telemetry.accelerationZ),
  );

  const thrust = calculateAssumedThrust(telemetry);

  // Newton's second law:
  // F_aero = m*a - F_thrust - F_gravity
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
 * Invert the quadratic drag equation to find an airflow that could have
 * produced the observed user-driven motion:
 *
 *   F_drag = 1/2 * rho * Cd * A * |Vrel|^2
 *   Vrel   = sqrt(2 * |F_drag| / (rho * Cd * A))
 *   Vwind  = Vdrone + Vrel
 *
 * The relative airflow points in the same direction as the aerodynamic force
 * on the vehicle. This is one plausible solution, not a unique reconstruction.
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

  if (aerodynamicForce.magnitude >= PHYSICS.minimumForceN) {
    const dragArea = PHYSICS.dragCoefficient * PHYSICS.referenceAreaM2;
    const relativeSpeed = Math.sqrt(
      (2 * aerodynamicForce.magnitude) /
        (PHYSICS.airDensityKgM3 * dragArea),
    );

    relativeAirflow
      .set(aerodynamicForce.x, aerodynamicForce.y, aerodynamicForce.z)
      .normalize()
      .multiplyScalar(relativeSpeed);

    inferredWind.add(relativeAirflow);
  }

  if (inferredWind.length() > PHYSICS.maximumWindSpeedMps) {
    inferredWind.setLength(PHYSICS.maximumWindSpeedMps);
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
 * Turn the single inferred wind vector into a display field. A potential-flow
 * sphere approximation bends air around the body, while a Gaussian velocity
 * deficit forms a widening wake downstream. Both effects are driven entirely
 * by the inverse-physics wind estimate.
 */
export function predictWind(
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
  if (speed < 0.02) {
    return modeledFlow.set(0, 0, 0);
  }

  modeledFlow.copy(baseFlow);
  flowDirection.copy(baseFlow).normalize();
  relativePosition.copy(samplePosition).sub(dronePosition);

  const distance = relativePosition.length();
  const bodyRadius = 1.05;

  if (distance > bodyRadius) {
    radialDirection.copy(relativePosition).divideScalar(distance);

    // Inviscid potential flow around a sphere. It is used only as a compact
    // visual approximation for near-body deflection.
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
    if (
      !params.steadyDirection &&
      lateralDirection.lengthSq() > 1e-8
    ) {
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

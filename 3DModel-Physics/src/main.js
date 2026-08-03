import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import "./style.css";
import {
  inferWindFromMotion,
  PHYSICS,
  predictWind,
} from "./windModel.js";
import {
  createNumericMotionPlan,
  sampleNumericMotion,
} from "./numericMotion.js";

const canvas = document.querySelector("#scene");
const app = document.querySelector("#app");

const scene = new THREE.Scene();
scene.fog = new THREE.Fog(0x07111d, 19, 39);

const camera = new THREE.PerspectiveCamera(48, 1, 0.1, 100);
camera.position.set(12, 11, 15);

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setClearColor(0x07111d, 1);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const orbitControls = new OrbitControls(camera, renderer.domElement);
orbitControls.enableDamping = true;
orbitControls.target.set(0, 1.2, 0);
orbitControls.maxPolarAngle = Math.PI * 0.47;
orbitControls.minDistance = 8;
orbitControls.maxDistance = 32;

scene.add(new THREE.HemisphereLight(0xccecff, 0x18202b, 2.2));

const sun = new THREE.DirectionalLight(0xffffff, 2.7);
sun.position.set(7, 14, 9);
sun.castShadow = true;
sun.shadow.mapSize.set(2048, 2048);
scene.add(sun);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(24, 24),
  new THREE.MeshStandardMaterial({
    color: 0x0a1724,
    roughness: 0.92,
    metalness: 0.05,
  }),
);
floor.rotation.x = -Math.PI / 2;
floor.receiveShadow = true;
scene.add(floor);

const grid = new THREE.GridHelper(24, 24, 0x4bb5e8, 0x213d52);
grid.material.opacity = 0.43;
grid.material.transparent = true;
scene.add(grid);

const ORIGIN_ALTITUDE = 1.35;

const telemetry = {
  yawDegrees: 0,
  pitchDegrees: 0,
  rollDegrees: 0,
  altitude: ORIGIN_ALTITUDE,
  velocityX: 0,
  velocityY: 0,
  velocityZ: 0,
  speed: 0,
  accelerationX: 0,
  accelerationY: 0,
  accelerationZ: 0,
  accelerationMagnitude: 0,
  estimatedWindX: 0,
  estimatedWindY: 0,
  estimatedWindZ: 0,
  estimatedWindSpeed: 0,
  estimatedWindConfidence: 0,
  aerodynamicForceX: 0,
  aerodynamicForceY: 0,
  aerodynamicForceZ: 0,
  aerodynamicForceMagnitude: 0,
};

const params = {};
let droneHeight = telemetry.altitude;
const DRAG_LIMIT = 9.5;

function createDrone() {
  const group = new THREE.Group();
  group.name = "drone";

  const bodyMaterial = new THREE.MeshStandardMaterial({
    color: 0xe9f6ff,
    roughness: 0.28,
    metalness: 0.5,
  });
  const darkMaterial = new THREE.MeshStandardMaterial({
    color: 0x182431,
    roughness: 0.45,
    metalness: 0.42,
  });
  const accentMaterial = new THREE.MeshStandardMaterial({
    color: 0x4ec5ff,
    emissive: 0x0b5273,
    emissiveIntensity: 0.55,
    roughness: 0.3,
  });

  const body = new THREE.Mesh(
    new THREE.BoxGeometry(1.55, 0.42, 1.1),
    bodyMaterial,
  );
  body.scale.set(1, 0.78, 1);
  body.castShadow = true;
  group.add(body);

  const nose = new THREE.Mesh(
    new THREE.ConeGeometry(0.35, 0.75, 20),
    accentMaterial,
  );
  nose.rotation.z = -Math.PI / 2;
  nose.position.x = 1.02;
  nose.castShadow = true;
  group.add(nose);

  const armGeometry = new THREE.BoxGeometry(2.9, 0.12, 0.16);
  for (const angle of [Math.PI / 4, -Math.PI / 4]) {
    const arm = new THREE.Mesh(armGeometry, darkMaterial);
    arm.rotation.y = angle;
    arm.castShadow = true;
    group.add(arm);
  }

  const rotorPositions = [
    [1.04, 0.13, 1.04],
    [1.04, 0.13, -1.04],
    [-1.04, 0.13, 1.04],
    [-1.04, 0.13, -1.04],
  ];

  for (const [x, y, z] of rotorPositions) {
    const motor = new THREE.Mesh(
      new THREE.CylinderGeometry(0.16, 0.2, 0.26, 20),
      darkMaterial,
    );
    motor.position.set(x, y, z);
    motor.castShadow = true;
    group.add(motor);

    const rotor = new THREE.Mesh(
      new THREE.CylinderGeometry(0.55, 0.55, 0.025, 34),
      new THREE.MeshStandardMaterial({
        color: 0x7fd7ff,
        transparent: true,
        opacity: 0.38,
        roughness: 0.25,
      }),
    );
    rotor.position.set(x, y + 0.18, z);
    rotor.castShadow = true;
    rotor.userData.isRotor = true;
    group.add(rotor);
  }

  const landingBarGeometry = new THREE.BoxGeometry(1.55, 0.08, 0.08);
  for (const z of [-0.46, 0.46]) {
    const landingBar = new THREE.Mesh(landingBarGeometry, darkMaterial);
    landingBar.position.set(-0.08, -0.42, z);
    group.add(landingBar);
  }

  group.position.set(0, droneHeight, 0);
  group.rotation.order = "YZX";
  return group;
}

const drone = createDrone();
scene.add(drone);

const droneMeshes = [];
drone.traverse((object) => {
  if (object.isMesh) droneMeshes.push(object);
});

const arrowGroup = new THREE.Group();
scene.add(arrowGroup);

const arrowSamples = [];
const horizontalOffsets = [-3, -2, -1, 0, 1, 2, 3];
const verticalOffsets = [-2, -1, 0, 1, 2];

for (const x of horizontalOffsets) {
  for (const y of verticalOffsets) {
    for (const z of horizontalOffsets) {
      if (Math.hypot(x, y, z) < 1.25) continue;

      const arrow = new THREE.ArrowHelper(
        new THREE.Vector3(1, 0, 0),
        new THREE.Vector3(),
        1,
        0x52b9f5,
        0.28,
        0.13,
      );
      arrowGroup.add(arrow);
      arrowSamples.push({
        localOffset: new THREE.Vector3(x, y, z),
        arrow,
      });
    }
  }
}

const localWindMarker = new THREE.ArrowHelper(
  new THREE.Vector3(1, 0, 0),
  new THREE.Vector3(),
  2,
  0xffffff,
  0.42,
  0.2,
);
scene.add(localWindMarker);

const color = new THREE.Color();
const direction = new THREE.Vector3();
const samplePosition = new THREE.Vector3();

function speedColor(speed) {
  const normalized = THREE.MathUtils.clamp(speed / 15, 0, 1);
  color.setHSL(0.61 - normalized * 0.61, 0.88, 0.56);
  return color;
}

function updateWindField(timeSeconds) {
  for (const sample of arrowSamples) {
    samplePosition.copy(drone.position).add(sample.localOffset);
    samplePosition.y = Math.max(samplePosition.y, 0.2);

    const wind = predictWind(
      samplePosition,
      drone.position,
      timeSeconds,
      params,
      telemetry,
    );
    const speed = wind.length();

    sample.arrow.position.copy(samplePosition);
    sample.arrow.visible = speed > 0.02;
    if (!sample.arrow.visible) continue;

    direction.copy(wind).normalize();
    sample.arrow.setDirection(direction);
    sample.arrow.setLength(
      THREE.MathUtils.clamp(0.32 + speed * 0.25, 0.5, 3.4),
      0.3,
      0.14,
    );
    sample.arrow.setColor(speedColor(speed));
  }

  const inferredWind = direction.set(
    telemetry.estimatedWindX,
    telemetry.estimatedWindY,
    telemetry.estimatedWindZ,
  );
  const inferredSpeed = inferredWind.length();

  localWindMarker.visible = inferredSpeed > 0.02;
  if (localWindMarker.visible) {
    localWindMarker.position.copy(drone.position);
    localWindMarker.position.y += 1.05;
    localWindMarker.setDirection(inferredWind.normalize());
    localWindMarker.setLength(
      THREE.MathUtils.clamp(0.55 + inferredSpeed * 0.35, 1, 4.5),
      0.42,
      0.2,
    );
    localWindMarker.setColor(speedColor(inferredSpeed));
  }
}

const previousDronePosition = drone.position.clone();
const measuredVelocity = new THREE.Vector3();
const smoothedVelocity = new THREE.Vector3();
const previousVelocity = new THREE.Vector3();
const measuredAcceleration = new THREE.Vector3();
const smoothedAcceleration = new THREE.Vector3();
const smoothedWind = new THREE.Vector3();
const targetWind = new THREE.Vector3();

const programmedMotion = {
  active: false,
  holdingResult: false,
  plan: null,
  elapsedSeconds: 0,
  peakForwardScore: -Infinity,
  peakWind: new THREE.Vector3(),
  peakForce: { x: 0, y: 0, z: 0, magnitude: 0 },
  travelDirection: new THREE.Vector3(),
};

let dragging = false;
let hovered = false;

function updateMotion(deltaTime) {
  if (deltaTime <= 0) return;

  measuredVelocity
    .copy(drone.position)
    .sub(previousDronePosition)
    .divideScalar(deltaTime);

  if (measuredVelocity.length() > 10) measuredVelocity.setLength(10);
  if (!dragging) measuredVelocity.set(0, 0, 0);

  smoothedVelocity.lerp(measuredVelocity, 1 - Math.exp(-9 * deltaTime));
  if (!dragging) smoothedVelocity.multiplyScalar(Math.exp(-9 * deltaTime));

  measuredAcceleration
    .copy(smoothedVelocity)
    .sub(previousVelocity)
    .divideScalar(deltaTime);
  if (measuredAcceleration.length() > 30) measuredAcceleration.setLength(30);

  smoothedAcceleration.lerp(
    measuredAcceleration,
    1 - Math.exp(-7 * deltaTime),
  );
  if (!dragging) smoothedAcceleration.multiplyScalar(Math.exp(-10 * deltaTime));

  telemetry.velocityX = smoothedVelocity.x;
  telemetry.velocityY = smoothedVelocity.y;
  telemetry.velocityZ = smoothedVelocity.z;
  telemetry.speed = smoothedVelocity.length();
  telemetry.accelerationX = smoothedAcceleration.x;
  telemetry.accelerationY = smoothedAcceleration.y;
  telemetry.accelerationZ = smoothedAcceleration.z;
  telemetry.accelerationMagnitude = smoothedAcceleration.length();

  previousDronePosition.copy(drone.position);
  previousVelocity.copy(smoothedVelocity);
}

function resetKinematics() {
  measuredVelocity.set(0, 0, 0);
  smoothedVelocity.set(0, 0, 0);
  previousVelocity.set(0, 0, 0);
  measuredAcceleration.set(0, 0, 0);
  smoothedAcceleration.set(0, 0, 0);
  telemetry.velocityX = 0;
  telemetry.velocityY = 0;
  telemetry.velocityZ = 0;
  telemetry.speed = 0;
  telemetry.accelerationX = 0;
  telemetry.accelerationY = 0;
  telemetry.accelerationZ = 0;
  telemetry.accelerationMagnitude = 0;
  previousDronePosition.copy(drone.position);
}

function updateProgrammedMotion(elapsedDelta) {
  if (!programmedMotion.active || !programmedMotion.plan) return;

  programmedMotion.elapsedSeconds += elapsedDelta;
  const sample = sampleNumericMotion(
    programmedMotion.plan,
    programmedMotion.elapsedSeconds,
  );

  drone.position.set(
    sample.position.x,
    sample.position.y,
    sample.position.z,
  );
  droneHeight = sample.position.y;
  telemetry.altitude = droneHeight;
  dragPlane.constant = -droneHeight;

  applyDroneAttitude(
    sample.attitude.yaw,
    sample.attitude.pitch,
    sample.attitude.roll,
  );

  telemetry.velocityX = sample.velocity.x;
  telemetry.velocityY = sample.velocity.y;
  telemetry.velocityZ = sample.velocity.z;
  telemetry.speed = Math.hypot(
    sample.velocity.x,
    sample.velocity.y,
    sample.velocity.z,
  );
  telemetry.accelerationX = sample.acceleration.x;
  telemetry.accelerationY = sample.acceleration.y;
  telemetry.accelerationZ = sample.acceleration.z;
  telemetry.accelerationMagnitude = Math.hypot(
    sample.acceleration.x,
    sample.acceleration.y,
    sample.acceleration.z,
  );

  numericProgress.value = sample.progress;
  numericStatus.classList.remove("error");
  numericStatus.textContent =
    `Animating ${(sample.progress * 100).toFixed(0)}% — ` +
    `${sample.elapsedSeconds.toFixed(2)} / ` +
    `${programmedMotion.plan.durationSeconds.toFixed(2)} s`;

  if (sample.done) {
    programmedMotion.active = false;
    programmedMotion.holdingResult = true;
    runNumericButton.disabled = false;
    runNumericButton.textContent = "Animate again";
    numericStatus.textContent =
      `Arrived in ${programmedMotion.plan.durationSeconds.toFixed(2)} s. ` +
      "Holding the strongest forward inferred wind for inspection.";
  }
}

function updatePhysicsEstimate(deltaTime) {
  const estimate = inferWindFromMotion(telemetry);
  targetWind.set(estimate.x, estimate.y, estimate.z);
  let displayedForce = estimate.aerodynamicForce;
  const numericMode = numericModeInput.checked;

  if (
    numericMode &&
    programmedMotion.holdingResult &&
    programmedMotion.peakWind.lengthSq() > 0
  ) {
    targetWind.copy(programmedMotion.peakWind);
    displayedForce = programmedMotion.peakForce;
  }

  const hasAttitudeDemand =
    Math.abs(telemetry.pitchDegrees) > 0.5 ||
    Math.abs(telemetry.rollDegrees) > 0.5;

  // Releasing the pointer is not a physical braking maneuver. Decay the last
  // estimate instead of interpreting the synthetic velocity decay as a gust
  // in the opposite direction.
  if (!numericMode && !dragging && !hasAttitudeDemand) {
    targetWind.set(0, 0, 0);
  }

  if (
    numericMode &&
    !programmedMotion.active &&
    !programmedMotion.holdingResult
  ) {
    targetWind.set(0, 0, 0);
  }

  if (numericMode && programmedMotion.active) {
    const forwardScore = programmedMotion.travelDirection.lengthSq() > 0
      ? targetWind.dot(programmedMotion.travelDirection)
      : targetWind.length();

    if (
      forwardScore > programmedMotion.peakForwardScore &&
      targetWind.lengthSq() > 0.0004
    ) {
      programmedMotion.peakForwardScore = forwardScore;
      programmedMotion.peakWind.copy(targetWind);
      programmedMotion.peakForce = { ...estimate.aerodynamicForce };
    }
  }

  const hasDrivenInput =
    dragging ||
    hasAttitudeDemand ||
    programmedMotion.active ||
    programmedMotion.holdingResult;
  const smoothingRate = hasDrivenInput ? 7 : 3;
  smoothedWind.lerp(targetWind, 1 - Math.exp(-smoothingRate * deltaTime));

  telemetry.estimatedWindX = smoothedWind.x;
  telemetry.estimatedWindY = smoothedWind.y;
  telemetry.estimatedWindZ = smoothedWind.z;
  telemetry.estimatedWindSpeed = smoothedWind.length();
  telemetry.aerodynamicForceX = displayedForce.x;
  telemetry.aerodynamicForceY = displayedForce.y;
  telemetry.aerodynamicForceZ = displayedForce.z;
  telemetry.aerodynamicForceMagnitude = displayedForce.magnitude;

  const signal = THREE.MathUtils.clamp(
    (telemetry.speed + telemetry.accelerationMagnitude * 0.2) / 5,
    0,
    1,
  );
  if (numericMode && programmedMotion.holdingResult) {
    telemetry.estimatedWindConfidence = 100;
  } else {
    telemetry.estimatedWindConfidence =
      hasDrivenInput ? 35 + signal * 65 : 0;
  }
}

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const dragPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -droneHeight);
const dragPoint = new THREE.Vector3();
const dragOffset = new THREE.Vector3();

function updatePointer(event) {
  const bounds = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - bounds.left) / bounds.width) * 2 - 1;
  pointer.y = -((event.clientY - bounds.top) / bounds.height) * 2 + 1;
}

function isDroneHit() {
  raycaster.setFromCamera(pointer, camera);
  return raycaster.intersectObjects(droneMeshes, false).length > 0;
}

renderer.domElement.addEventListener("pointerdown", (event) => {
  if (numericModeInput.checked) return;
  updatePointer(event);
  if (!isDroneHit()) return;

  dragging = true;
  orbitControls.enabled = false;
  renderer.domElement.setPointerCapture(event.pointerId);
  raycaster.ray.intersectPlane(dragPlane, dragPoint);
  dragOffset.copy(drone.position).sub(dragPoint);
});

renderer.domElement.addEventListener("pointermove", (event) => {
  if (numericModeInput.checked) {
    renderer.domElement.style.cursor = "default";
    return;
  }
  updatePointer(event);

  if (!dragging) {
    hovered = isDroneHit();
    renderer.domElement.style.cursor = hovered ? "grab" : "default";
    return;
  }

  raycaster.setFromCamera(pointer, camera);
  if (raycaster.ray.intersectPlane(dragPlane, dragPoint)) {
    drone.position.copy(dragPoint).add(dragOffset);
    drone.position.x = THREE.MathUtils.clamp(drone.position.x, -DRAG_LIMIT, DRAG_LIMIT);
    drone.position.z = THREE.MathUtils.clamp(drone.position.z, -DRAG_LIMIT, DRAG_LIMIT);
    drone.position.y = droneHeight;
  }
  renderer.domElement.style.cursor = "grabbing";
});

function stopDragging(event) {
  if (!dragging) return;
  dragging = false;
  orbitControls.enabled = true;
  renderer.domElement.style.cursor = hovered ? "grab" : "default";
  if (renderer.domElement.hasPointerCapture(event.pointerId)) {
    renderer.domElement.releasePointerCapture(event.pointerId);
  }
}

renderer.domElement.addEventListener("pointerup", stopDragging);
renderer.domElement.addEventListener("pointercancel", stopDragging);

function applyDroneAttitude(yaw, pitch, roll) {
  telemetry.yawDegrees = yaw;
  telemetry.pitchDegrees = pitch;
  telemetry.rollDegrees = roll;
  drone.rotation.y = -THREE.MathUtils.degToRad(yaw);
  drone.rotation.z = THREE.MathUtils.degToRad(pitch);
  drone.rotation.x = THREE.MathUtils.degToRad(roll);
}

function connectAttitudeControl(id, outputId, property, axis) {
  const input = document.querySelector(id);
  const output = document.querySelector(outputId);

  const update = () => {
    const degrees = Number(input.value);
    telemetry[property] = degrees;
    output.textContent = `${degrees.toFixed(0)}°`;

    if (axis === "yaw") drone.rotation.y = -THREE.MathUtils.degToRad(degrees);
    if (axis === "pitch") drone.rotation.z = THREE.MathUtils.degToRad(degrees);
    if (axis === "roll") drone.rotation.x = THREE.MathUtils.degToRad(degrees);
  };

  input.addEventListener("input", update);
  update();
}

connectAttitudeControl("#yaw", "#yaw-output", "yawDegrees", "yaw");
connectAttitudeControl("#pitch", "#pitch-output", "pitchDegrees", "pitch");
connectAttitudeControl("#roll", "#roll-output", "rollDegrees", "roll");

const altitudeInput = document.querySelector("#altitude");
const altitudeOutput = document.querySelector("#altitude-output");

function updateAltitude() {
  droneHeight = Number(altitudeInput.value);
  telemetry.altitude = droneHeight;
  drone.position.y = droneHeight;
  dragPlane.constant = -droneHeight;
  altitudeOutput.textContent = `${droneHeight.toFixed(2)} m`;
}

altitudeInput.addEventListener("input", updateAltitude);
updateAltitude();

const numericModeInput = document.querySelector("#numeric-input-mode");
const numericControls = document.querySelector("#numeric-motion-controls");
const manualControls = document.querySelector("#manual-motion-controls");
const numericProgress = document.querySelector("#numeric-progress");
const numericStatus = document.querySelector("#numeric-status");
const runNumericButton = document.querySelector("#run-numeric-motion");
const cancelNumericButton = document.querySelector("#cancel-numeric-motion");

const numericInputs = {
  x: document.querySelector("#numeric-x"),
  y: document.querySelector("#numeric-y"),
  z: document.querySelector("#numeric-z"),
  roll: document.querySelector("#numeric-roll"),
  pitch: document.querySelector("#numeric-pitch"),
  yaw: document.querySelector("#numeric-yaw"),
  duration: document.querySelector("#numeric-duration"),
};

function cancelNumericMotion(message = "Result cleared.") {
  programmedMotion.active = false;
  programmedMotion.holdingResult = false;
  programmedMotion.plan = null;
  programmedMotion.elapsedSeconds = 0;
  programmedMotion.peakForwardScore = -Infinity;
  programmedMotion.peakWind.set(0, 0, 0);
  programmedMotion.peakForce = { x: 0, y: 0, z: 0, magnitude: 0 };
  programmedMotion.travelDirection.set(0, 0, 0);
  smoothedWind.set(0, 0, 0);
  resetKinematics();
  numericProgress.value = 0;
  numericStatus.classList.remove("error");
  numericStatus.textContent = message;
  runNumericButton.disabled = false;
  runNumericButton.textContent = "Animate motion";
}

function readNumericValue(input, label) {
  const value = Number(input.value);
  if (!Number.isFinite(value)) {
    throw new Error(`${label} must be a number.`);
  }
  return value;
}

function validatePlanTarget(plan) {
  const { x, y, z } = plan.targetPosition;
  if (Math.abs(x) > DRAG_LIMIT || Math.abs(z) > DRAG_LIMIT) {
    throw new Error(`Target x and z must remain within ±${DRAG_LIMIT} m.`);
  }
  if (y < 0.5 || y > 6) {
    throw new Error("Target y must remain between 0.5 m and 6 m.");
  }
  if (
    Math.abs(plan.targetAttitude.roll) > 30 ||
    Math.abs(plan.targetAttitude.pitch) > 30
  ) {
    throw new Error("Target roll and pitch must remain within ±30°.");
  }
}

function startNumericMotion() {
  try {
    const plan = createNumericMotionPlan({
      startPosition: drone.position,
      offsetInches: {
        x: readNumericValue(numericInputs.x, "X offset"),
        y: readNumericValue(numericInputs.y, "Y offset"),
        z: readNumericValue(numericInputs.z, "Z offset"),
      },
      startAttitude: {
        roll: telemetry.rollDegrees,
        pitch: telemetry.pitchDegrees,
        yaw: telemetry.yawDegrees,
      },
      attitudeOffsetDegrees: {
        roll: readNumericValue(numericInputs.roll, "Roll offset"),
        pitch: readNumericValue(numericInputs.pitch, "Pitch offset"),
        yaw: readNumericValue(numericInputs.yaw, "Yaw offset"),
      },
      durationSeconds: readNumericValue(numericInputs.duration, "Travel time"),
    });

    validatePlanTarget(plan);
    programmedMotion.active = true;
    programmedMotion.holdingResult = false;
    programmedMotion.plan = plan;
    programmedMotion.elapsedSeconds = 0;
    programmedMotion.peakForwardScore = -Infinity;
    programmedMotion.peakWind.set(0, 0, 0);
    programmedMotion.peakForce = { x: 0, y: 0, z: 0, magnitude: 0 };
    programmedMotion.travelDirection.set(
      plan.offsetMeters.x,
      plan.offsetMeters.y,
      plan.offsetMeters.z,
    );
    if (programmedMotion.travelDirection.lengthSq() > 0) {
      programmedMotion.travelDirection.normalize();
    }

    resetKinematics();
    smoothedWind.set(0, 0, 0);
    numericProgress.value = 0;
    numericStatus.classList.remove("error");
    numericStatus.textContent =
      `Animating ${(plan.distanceMeters / 0.0254).toFixed(1)} in ` +
      `over ${plan.durationSeconds.toFixed(2)} s.`;
    runNumericButton.disabled = true;
    runNumericButton.textContent = "Animating…";
  } catch (error) {
    numericStatus.classList.add("error");
    numericStatus.textContent = error.message;
  }
}

function synchronizeManualControls() {
  const normalizedYaw =
    ((telemetry.yawDegrees + 180) % 360 + 360) % 360 - 180;
  const values = [
    [altitudeInput, drone.position.y, altitudeOutput, `${drone.position.y.toFixed(2)} m`],
    [document.querySelector("#yaw"), normalizedYaw, document.querySelector("#yaw-output"), `${normalizedYaw.toFixed(0)}°`],
    [document.querySelector("#pitch"), telemetry.pitchDegrees, document.querySelector("#pitch-output"), `${telemetry.pitchDegrees.toFixed(0)}°`],
    [document.querySelector("#roll"), telemetry.rollDegrees, document.querySelector("#roll-output"), `${telemetry.rollDegrees.toFixed(0)}°`],
  ];

  for (const [input, value, output, label] of values) {
    input.value = String(value);
    output.textContent = label;
  }
  telemetry.yawDegrees = normalizedYaw;
  droneHeight = drone.position.y;
  dragPlane.constant = -droneHeight;
  previousDronePosition.copy(drone.position);
}

function updateInputMode() {
  const numericMode = numericModeInput.checked;
  dragging = false;
  orbitControls.enabled = true;
  cancelNumericMotion(
    numericMode ? "Ready for numeric input." : "Numeric result cleared.",
  );
  numericControls.hidden = !numericMode;
  manualControls.hidden = numericMode;
  renderer.domElement.style.cursor = numericMode ? "default" : "grab";
  if (!numericMode) synchronizeManualControls();
}

numericModeInput.addEventListener("change", updateInputMode);
runNumericButton.addEventListener("click", startNumericMotion);
cancelNumericButton.addEventListener("click", () => cancelNumericMotion());
updateInputMode();

function resetDroneToOrigin() {
  cancelNumericMotion(
    "Drone reset to the origin at 1.35 m with a level attitude.",
  );

  droneHeight = ORIGIN_ALTITUDE;
  drone.position.set(0, ORIGIN_ALTITUDE, 0);
  telemetry.altitude = ORIGIN_ALTITUDE;
  dragPlane.constant = -ORIGIN_ALTITUDE;
  applyDroneAttitude(0, 0, 0);
  resetKinematics();

  altitudeInput.value = String(ORIGIN_ALTITUDE);
  altitudeOutput.textContent = `${ORIGIN_ALTITUDE.toFixed(2)} m`;
  for (const [inputId, outputId] of [
    ["#yaw", "#yaw-output"],
    ["#pitch", "#pitch-output"],
    ["#roll", "#roll-output"],
  ]) {
    document.querySelector(inputId).value = "0";
    document.querySelector(outputId).textContent = "0°";
  }

  camera.position.set(12, 11, 15);
  orbitControls.target.set(0, 1.2, 0);
  orbitControls.update();
}

document
  .querySelector("#reset-button")
  .addEventListener("click", resetDroneToOrigin);

const readouts = {
  x: document.querySelector("#x-value"),
  y: document.querySelector("#y-value"),
  z: document.querySelector("#z-value"),
  mode: document.querySelector("#mode-value"),
  droneSpeed: document.querySelector("#drone-speed-value"),
  acceleration: document.querySelector("#drone-acceleration-value"),
  windSpeed: document.querySelector("#estimated-wind-value"),
  windComponents: document.querySelector("#estimated-wind-components"),
  force: document.querySelector("#aerodynamic-force-value"),
  forceComponents: document.querySelector("#aerodynamic-force-components"),
  confidence: document.querySelector("#wind-confidence-value"),
  confidenceBar: document.querySelector("#wind-confidence-bar"),
};

function updateReadouts() {
  readouts.x.textContent = `${drone.position.x.toFixed(2)} m`;
  readouts.y.textContent = `${drone.position.y.toFixed(2)} m`;
  readouts.z.textContent = `${drone.position.z.toFixed(2)} m`;
  readouts.mode.textContent = numericModeInput.checked ? "Numeric" : "Drag";
  readouts.droneSpeed.textContent = `${telemetry.speed.toFixed(2)} m/s`;
  readouts.acceleration.textContent = `${telemetry.accelerationMagnitude.toFixed(2)} m/s²`;
  readouts.windSpeed.textContent = `${telemetry.estimatedWindSpeed.toFixed(2)} m/s`;
  readouts.windComponents.textContent =
    `u: ${telemetry.estimatedWindX.toFixed(2)}, ` +
    `v: ${telemetry.estimatedWindZ.toFixed(2)}, ` +
    `w: ${telemetry.estimatedWindY.toFixed(2)} m/s`;
  readouts.force.textContent = `${telemetry.aerodynamicForceMagnitude.toFixed(2)} N`;
  readouts.forceComponents.textContent =
    `Fx: ${telemetry.aerodynamicForceX.toFixed(2)}, ` +
    `Fy: ${telemetry.aerodynamicForceY.toFixed(2)}, ` +
    `Fz: ${telemetry.aerodynamicForceZ.toFixed(2)} N`;
  readouts.confidence.textContent = `${telemetry.estimatedWindConfidence.toFixed(0)}%`;
  readouts.confidenceBar.style.width = `${telemetry.estimatedWindConfidence}%`;
}

function resize() {
  const width = app.clientWidth;
  const height = app.clientHeight;
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
  renderer.setSize(width, height);
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
}

window.addEventListener("resize", resize);
resize();

const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);
  const rawDeltaTime = clock.getDelta();
  const deltaTime = Math.min(rawDeltaTime, 0.1);
  const timeSeconds = clock.elapsedTime;

  if (numericModeInput.checked) {
    updateProgrammedMotion(rawDeltaTime);
  } else {
    updateMotion(deltaTime);
  }
  updatePhysicsEstimate(deltaTime);
  updateWindField(timeSeconds);
  updateReadouts();

  for (const object of drone.children) {
    if (object.userData.isRotor) object.rotation.y += deltaTime * 24;
  }

  orbitControls.update();
  renderer.render(scene, camera);
}

console.info(
  `Inverse-physics model: m=${PHYSICS.massKg} kg, rho=${PHYSICS.airDensityKgM3} kg/m³`,
);
animate();

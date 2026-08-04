import { createMlTelemetrySample } from "./mlTelemetry.js";
import { sampleNumericMotion } from "./numericMotion.js";

export function createTrajectorySampleTimes(
  durationSeconds,
  { targetIntervalSeconds = 0.1, maximumSamples = 121 } = {},
) {
  const duration = Number(durationSeconds);
  if (!Number.isFinite(duration) || duration <= 0) {
    throw new Error("Trajectory duration must be positive.");
  }
  const requestedSteps = Math.max(
    1,
    Math.ceil(duration / targetIntervalSeconds),
  );
  const steps = Math.min(requestedSteps, maximumSamples - 1);
  return Array.from(
    { length: steps + 1 },
    (_, index) => duration * index / steps,
  );
}

export async function prepareMlTrajectory({
  plan,
  sessionId,
  batteryVoltage,
  batteryCurrent,
  fetchPredictions,
  onProgress = () => {},
  isCancelled = () => false,
  sampling,
}) {
  const sampleTimes = createTrajectorySampleTimes(
    plan.durationSeconds,
    sampling,
  );
  const telemetry = sampleTimes.map((elapsedSeconds) => {
    const motion = sampleNumericMotion(plan, elapsedSeconds);
    return (
      createMlTelemetrySample({
        sessionId,
        elapsedSeconds,
        position: motion.position,
        attitudeDegrees: motion.attitude,
        batteryVoltage,
        batteryCurrent,
      })
    );
  });
  if (isCancelled()) return null;
  onProgress(0, sampleTimes.length);
  const predictions = await fetchPredictions(telemetry);
  if (isCancelled()) return null;
  if (!Array.isArray(predictions) || predictions.length !== sampleTimes.length) {
    throw new Error(
      `ML service returned ${predictions?.length ?? 0} predictions for ` +
      `${sampleTimes.length} trajectory samples.`,
    );
  }
  onProgress(sampleTimes.length, sampleTimes.length);
  return sampleTimes.map((elapsedSeconds, index) => ({
    elapsedSeconds,
    prediction: predictions[index],
  }));
}

function interpolateNumber(left, right, blend) {
  const a = Number(left);
  const b = Number(right);
  if (!Number.isFinite(a)) return Number.isFinite(b) ? b : 0;
  if (!Number.isFinite(b)) return a;
  return a + (b - a) * blend;
}

export function sampleMlTrajectory(predictions, elapsedSeconds) {
  if (!Array.isArray(predictions) || predictions.length === 0) return null;
  const elapsed = Math.max(0, Number(elapsedSeconds) || 0);
  if (elapsed <= predictions[0].elapsedSeconds) {
    return { ...predictions[0].prediction };
  }
  const final = predictions[predictions.length - 1];
  if (elapsed >= final.elapsedSeconds) return { ...final.prediction };

  let upperIndex = 1;
  while (
    upperIndex < predictions.length &&
    predictions[upperIndex].elapsedSeconds < elapsed
  ) {
    upperIndex += 1;
  }
  const lower = predictions[upperIndex - 1];
  const upper = predictions[upperIndex];
  const span = upper.elapsedSeconds - lower.elapsedSeconds;
  const blend = span > 0 ? (elapsed - lower.elapsedSeconds) / span : 0;
  const u = interpolateNumber(lower.prediction.u, upper.prediction.u, blend);
  const v = interpolateNumber(lower.prediction.v, upper.prediction.v, blend);
  const w = interpolateNumber(lower.prediction.w, upper.prediction.w, blend);

  return {
    ...lower.prediction,
    u,
    v,
    w,
    speed: Math.hypot(u, v, w),
    direction_confidence: interpolateNumber(
      lower.prediction.direction_confidence,
      upper.prediction.direction_confidence,
      blend,
    ),
  };
}

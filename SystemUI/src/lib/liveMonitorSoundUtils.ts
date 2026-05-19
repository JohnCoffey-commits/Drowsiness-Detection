import type { LiveAlertKind } from "@/lib/liveMonitorAlertUtils";

type BrowserWindowWithAudioContext = Window &
  typeof globalThis & {
    webkitAudioContext?: typeof AudioContext;
  };

interface SoundStep {
  frequency: number;
  durationMs: number;
  gapAfterMs?: number;
  gain: number;
  type: OscillatorType;
}

interface SoundPattern {
  steps: SoundStep[];
}

export interface AudioContextResult {
  ok: boolean;
  context?: AudioContext;
  error?: string;
}

export interface SoundPlaybackResult {
  ok: boolean;
  playedAt?: number;
  error?: string;
}

function getAudioContextConstructor(): typeof AudioContext | undefined {
  if (typeof window === "undefined") {
    return undefined;
  }

  const audioWindow = window as BrowserWindowWithAudioContext;
  return audioWindow.AudioContext ?? audioWindow.webkitAudioContext;
}

export async function createAudioContextSafely(
  existingContext: AudioContext | null
): Promise<AudioContextResult> {
  const AudioContextConstructor = getAudioContextConstructor();

  if (!AudioContextConstructor) {
    return {
      ok: false,
      error: "This browser does not support Web Audio API sound output.",
    };
  }

  try {
    const context =
      existingContext && existingContext.state !== "closed"
        ? existingContext
        : new AudioContextConstructor();

    if (context.state === "suspended") {
      await context.resume();
    }

    return { ok: true, context };
  } catch (error) {
    return {
      ok: false,
      error:
        error instanceof Error
          ? error.message
          : "Audio output could not be initialized.",
    };
  }
}

export async function closeAudioContextSafely(
  context: AudioContext | null
): Promise<void> {
  if (!context || context.state === "closed") {
    return;
  }

  try {
    await context.close();
  } catch {
    // Closing is best-effort cleanup; short scheduled sounds are non-critical.
  }
}

export function getSoundPatternForAlertKind(kind: LiveAlertKind): SoundPattern {
  if (kind === "eye_warning") {
    return {
      steps: [
        {
          frequency: 440,
          durationMs: 150,
          gapAfterMs: 70,
          gain: 0.035,
          type: "sine",
        },
        {
          frequency: 560,
          durationMs: 150,
          gain: 0.032,
          type: "sine",
        },
      ],
    };
  }

  if (kind === "mouth_warning") {
    return {
      steps: [
        {
          frequency: 392,
          durationMs: 170,
          gapAfterMs: 55,
          gain: 0.032,
          type: "sine",
        },
        {
          frequency: 494,
          durationMs: 135,
          gain: 0.03,
          type: "sine",
        },
      ],
    };
  }

  if (kind === "high_confidence") {
    return {
      steps: [
        {
          frequency: 720,
          durationMs: 95,
          gapAfterMs: 55,
          gain: 0.07,
          type: "triangle",
        },
        {
          frequency: 920,
          durationMs: 95,
          gapAfterMs: 55,
          gain: 0.075,
          type: "triangle",
        },
        {
          frequency: 720,
          durationMs: 95,
          gapAfterMs: 55,
          gain: 0.07,
          type: "triangle",
        },
        {
          frequency: 980,
          durationMs: 130,
          gain: 0.08,
          type: "triangle",
        },
      ],
    };
  }

  return {
    steps: [
      {
        frequency: 260,
        durationMs: 100,
        gain: 0.025,
        type: "sine",
      },
    ],
  };
}

function getTestSoundPattern(): SoundPattern {
  return {
    steps: [
      {
        frequency: 620,
        durationMs: 90,
        gapAfterMs: 45,
        gain: 0.04,
        type: "sine",
      },
      {
        frequency: 820,
        durationMs: 90,
        gain: 0.035,
        type: "sine",
      },
    ],
  };
}

async function playSoundPattern(
  context: AudioContext,
  pattern: SoundPattern
): Promise<SoundPlaybackResult> {
  if (context.state === "closed") {
    return { ok: false, error: "Audio output is closed. Turn sound alerts on again." };
  }

  try {
    if (context.state === "suspended") {
      await context.resume();
    }

    let offsetSeconds = 0.02;

    pattern.steps.forEach((step) => {
      const oscillator = context.createOscillator();
      const gain = context.createGain();
      const startAt = context.currentTime + offsetSeconds;
      const endAt = startAt + step.durationMs / 1000;
      const releaseStart = Math.max(startAt, endAt - 0.025);

      oscillator.frequency.setValueAtTime(step.frequency, startAt);
      oscillator.type = step.type;
      gain.gain.setValueAtTime(0.0001, startAt);
      gain.gain.linearRampToValueAtTime(step.gain, startAt + 0.015);
      gain.gain.setValueAtTime(step.gain, releaseStart);
      gain.gain.linearRampToValueAtTime(0.0001, endAt);

      oscillator.connect(gain);
      gain.connect(context.destination);
      oscillator.start(startAt);
      oscillator.stop(endAt + 0.01);
      oscillator.onended = () => {
        oscillator.disconnect();
        gain.disconnect();
      };

      offsetSeconds += (step.durationMs + (step.gapAfterMs ?? 0)) / 1000;
    });

    return { ok: true, playedAt: Date.now() };
  } catch (error) {
    return {
      ok: false,
      error:
        error instanceof Error ? error.message : "Sound playback could not be started.",
    };
  }
}

export function playLiveMonitorAlertSound(
  context: AudioContext,
  kind: LiveAlertKind
): Promise<SoundPlaybackResult> {
  return playSoundPattern(context, getSoundPatternForAlertKind(kind));
}

export function playTestSound(context: AudioContext): Promise<SoundPlaybackResult> {
  return playSoundPattern(context, getTestSoundPattern());
}

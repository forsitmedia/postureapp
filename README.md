# Sit Happens

Live posture scoring in the browser. Your webcam, your machine, no uploads.

You slouch into your screen like a shrimp. This notices, scores it, and
interrupts you with a video you will want to avoid seeing again.

## Run it

No build step, no install. Just serve the folder:

```bash
python3 -m http.server 8787
```

Then open <http://localhost:8787>.

A local server is required — `file://` will not work, because the browser
blocks camera access and ES module imports on that protocol.

## How it works

Pose landmarks come from MediaPipe Tasks Vision (`pose_landmarker_lite`),
running entirely client-side via WebAssembly.

Scoring is **personalised** rather than threshold-based. Generic posture
thresholds fail because bodies, chairs and desk heights differ. Instead the
app captures two reference poses from you:

1. your best posture
2. your worst deliberate shrimp

It then builds a feature vector from each and projects your live pose onto
the axis between them:

| feature | meaning                                    |
| ------- | ------------------------------------------ |
| `neck`  | head height above shoulders, scale-invariant |
| `size`  | shoulder width, a proxy for leaning in     |
| `tilt`  | shoulder imbalance, one-sided collapse     |
| `off`   | head drift off centre                      |

`neck` and `size` drive the main 0–100 score, weighted by how much each
actually moved between your two calibration poses — so the app leans on
whichever signal is most informative *for you*. `tilt` and `off` apply
independent penalties on top.

Every feature is divided by shoulder width, which makes the score invariant
to how far you sit from the camera.

## Controls

- **Alert threshold** — the score below which you count as shrimping
- **Grace period** — how long you may shrimp before the video fires
- **Recalibrate** — recapture your two reference poses
- **Reset demo** — clear session stats

## Privacy

The video stream never leaves the browser. There is no backend, no
analytics and no network traffic beyond fetching the model on first load.

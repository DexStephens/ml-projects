#!/usr/bin/env python3
"""
Synthetic motion detection demo (step 2).

Generates a scene with a moving rectangle ("person"), adds noise, and runs
frame differencing via cv2.absdiff to learn how motion detection works.

Pipeline:
  blank frame -> draw rectangle -> add noise -> absdiff -> threshold -> contours
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import cv2
import numpy as np

# --- Configurable parameters ---
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
NUM_FRAMES = 60
RECT_WIDTH = 80
RECT_HEIGHT = 160
RECT_SPEED_X = 4
RECT_START_X = 20
RECT_Y = (FRAME_HEIGHT - RECT_HEIGHT) // 2
NOISE_STD = 12.0
DIFF_THRESHOLD = 25
MIN_CONTOUR_AREA = 500


@dataclass(frozen=True)
class GroundTruthBox:
    """Known rectangle position for the synthetic 'person'."""

    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


@dataclass
class MotionResult:
    """Output of one frame's motion detection pass."""

    frame_index: int
    motion_detected: bool
    detected_boxes: list[tuple[int, int, int, int]]
    total_motion_area: int


def make_blank_frame(width: int, height: int) -> np.ndarray:
    """Dark background scene (single channel for simplicity)."""
    return np.zeros((height, width), dtype=np.uint8)


def draw_person(frame: np.ndarray, box: GroundTruthBox, intensity: int = 220) -> None:
    """Draw the synthetic person as a bright rectangle."""
    cv2.rectangle(
        frame,
        (box.x, box.y),
        (box.x + box.w, box.y + box.h),
        intensity,
        thickness=-1,
    )


def add_noise(frame: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise and clip back to valid uint8 range."""
    noisy = frame.astype(np.float32) + rng.normal(0.0, std, frame.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def ground_truth_for_frame(frame_index: int) -> GroundTruthBox:
    """Rectangle moves horizontally; position is known exactly."""
    x = RECT_START_X + frame_index * RECT_SPEED_X
    if x + RECT_WIDTH > FRAME_WIDTH:
        x = FRAME_WIDTH - RECT_WIDTH
    return GroundTruthBox(x=x, y=RECT_Y, w=RECT_WIDTH, h=RECT_HEIGHT)


def generate_frame(frame_index: int, rng: np.random.Generator) -> tuple[np.ndarray, GroundTruthBox]:
    """Build one synthetic frame with known ground truth."""
    frame = make_blank_frame(FRAME_WIDTH, FRAME_HEIGHT)
    box = ground_truth_for_frame(frame_index)
    draw_person(frame, box)
    frame = add_noise(frame, NOISE_STD, rng)
    return frame, box


def detect_motion(
    previous: np.ndarray,
    current: np.ndarray,
    threshold: int,
    min_area: int,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]], int]:
    """
    Classic frame differencing: absdiff -> threshold -> find contours.

    Returns:
        binary mask, list of bounding boxes (x, y, w, h), total contour area
    """
    diff = cv2.absdiff(previous, current)
    _, mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes: list[tuple[int, int, int, int]] = []
    total_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        total_area += int(area)
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append((x, y, w, h))

    return mask, boxes, total_area


def boxes_overlap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:
    """True if two axis-aligned boxes intersect."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (ax + aw < bx or bx + bw < ax or ay + ah < by or by + bh < ay)


def summarize_frame(
    frame_index: int,
    ground_truth: GroundTruthBox,
    result: MotionResult,
) -> str:
    gt = ground_truth.as_tuple()
    matched = any(boxes_overlap(gt, det) for det in result.detected_boxes)
    status = "MATCH" if matched else "MISS" if result.motion_detected else "no motion"
    det_str = ", ".join(f"{b}" for b in result.detected_boxes) or "none"
    return (
        f"frame={frame_index:03d}  gt={gt}  motion={result.motion_detected}  "
        f"detected=[{det_str}]  area={result.total_motion_area}  {status}"
    )


def run_demo(show: bool = False, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    previous: np.ndarray | None = None

    print("Synthetic motion detection (cv2.absdiff)")
    print(
        f"  size={FRAME_WIDTH}x{FRAME_HEIGHT}  frames={NUM_FRAMES}  "
        f"threshold={DIFF_THRESHOLD}  min_area={MIN_CONTOUR_AREA}"
    )
    print()

    for i in range(NUM_FRAMES):
        current, gt = generate_frame(i, rng)

        if previous is None:
            print(f"frame={i:03d}  gt={gt.as_tuple()}  (baseline, no diff yet)")
            previous = current
            if show:
                vis = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
                cv2.rectangle(vis, (gt.x, gt.y), (gt.x + gt.w, gt.y + gt.h), (0, 255, 0), 2)
                cv2.imshow("synthetic_motion", vis)
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            continue

        mask, boxes, total_area = detect_motion(
            previous, current, DIFF_THRESHOLD, MIN_CONTOUR_AREA
        )
        result = MotionResult(
            frame_index=i,
            motion_detected=len(boxes) > 0,
            detected_boxes=boxes,
            total_motion_area=total_area,
        )
        print(summarize_frame(i, gt, result))

        if show:
            vis = cv2.cvtColor(current, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(vis, (gt.x, gt.y), (gt.x + gt.w, gt.y + gt.h), (0, 255, 0), 2)
            for x, y, w, h in boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
            diff_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            combined = np.hstack([vis, diff_vis])
            cv2.imshow("synthetic_motion", combined)
            if cv2.waitKey(30) & 0xFF == ord("q"):
                break

        previous = current

    if show:
        cv2.destroyAllWindows()

    print()
    print("Done. Green = ground truth, red = detected motion (with --show).")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic OpenCV motion detection learning demo.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display frames and diff mask (requires GUI). Press q to quit early.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible noise.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_demo(show=args.show, seed=args.seed)

from __future__ import annotations

from dataclasses import dataclass

import cv2

from app.models.base_model import BasePoseModel, Frame
from app.schemas.pose import PoseKeypoint, PoseResult


# A compact skeleton subset for cleaner overlays.
SKELETON_EDGES = [
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 12),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]


@dataclass
class _MediaPipeState:
    pose: object | None = None
    landmarks: object | None = None


class MediaPipePoseModel(BasePoseModel):
    name = "mediapipe"

    def __init__(self) -> None:
        self._state = _MediaPipeState()

    def load_model(self) -> None:
        try:
            import mediapipe as mp

             # Some builds don't expose `mp.solutions` at top-level.
            try:
                mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
            except AttributeError:
                from mediapipe.python.solutions import pose as mp_pose  # type: ignore

            self._state.landmarks = mp_pose.PoseLandmark
            self._state.pose = mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=0.4,
                min_tracking_confidence=0.4,
            )
        except ImportError:
            self._state.pose = None
            self._state.landmarks = None

    def infer(self, frame: Frame) -> PoseResult:
        if self._state.pose is None:
            return self._heuristic_pose(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._state.pose.process(rgb)
        if not result.pose_landmarks:
            return PoseResult()

        h, w = frame.shape[:2]
        keypoints: list[PoseKeypoint] = []
        valid_xy: list[tuple[float, float]] = []
        vis_sum = 0.0

        for i, lm in enumerate(result.pose_landmarks.landmark):
            x = float(lm.x * w)
            y = float(lm.y * h)
            conf = float(max(0.0, min(1.0, lm.visibility)))
            keypoints.append(PoseKeypoint(name=str(i), x=x, y=y, confidence=conf))
            if conf > 0.2:
                valid_xy.append((x, y))
            vis_sum += conf

        bbox = None
        if valid_xy:
            xs = [p[0] for p in valid_xy]
            ys = [p[1] for p in valid_xy]
            bbox = [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))]

        return PoseResult(
            keypoints=keypoints,
            confidence=vis_sum / max(1, len(keypoints)),
            bbox=bbox,
        )

    def visualize(self, frame: Frame, pose_result: PoseResult) -> Frame:
        rendered = frame.copy()
        if not pose_result.keypoints:
            return rendered

        points: dict[int, tuple[int, int, float]] = {}
        for kp in pose_result.keypoints:
            idx = int(kp.name) if kp.name.isdigit() else -1
            points[idx] = (int(kp.x), int(kp.y), kp.confidence)
            if kp.confidence > 0.2:
                cv2.circle(rendered, (int(kp.x), int(kp.y)), 3, (0, 255, 0), -1)

        for s, e in SKELETON_EDGES:
            if s not in points or e not in points:
                continue
            sx, sy, sc = points[s]
            ex, ey, ec = points[e]
            if min(sc, ec) < 0.2:
                continue
            cv2.line(rendered, (sx, sy), (ex, ey), (255, 140, 0), 2)

        if pose_result.bbox:
            x1, y1, x2, y2 = [int(v) for v in pose_result.bbox]
            cv2.rectangle(rendered, (x1, y1), (x2, y2), (50, 220, 255), 2)

        return rendered

    def _heuristic_pose(self, frame: Frame) -> PoseResult:
        h, w = frame.shape[:2]
        keypoints = [
            PoseKeypoint(name="11", x=0.42 * w, y=0.35 * h, confidence=0.25),
            PoseKeypoint(name="12", x=0.58 * w, y=0.35 * h, confidence=0.25),
            PoseKeypoint(name="23", x=0.45 * w, y=0.55 * h, confidence=0.2),
            PoseKeypoint(name="24", x=0.55 * w, y=0.55 * h, confidence=0.2),
        ]
        return PoseResult(
            keypoints=keypoints,
            confidence=0.23,
            bbox=[0.35 * w, 0.25 * h, 0.65 * w, 0.7 * h],
        )

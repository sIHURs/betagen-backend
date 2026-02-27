from __future__ import annotations

from pathlib import Path

import cv2

from app.models.registry import PoseModelRegistry
from app.schemas.pose import FramePoseRecord, KeypointsPayload, PoseResult, ProcessingConfig


class VideoProcessor:
    def __init__(self, registry: PoseModelRegistry, max_video_seconds: int) -> None:
        self.registry = registry
        self.max_video_seconds = max_video_seconds

    def process_video(
        self,
        video_id: str,
        input_path: Path,
        output_dir: Path,
        config: ProcessingConfig,
    ) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        keypoints_path = output_dir / "keypoints.json"
        overlay_path = output_dir / "overlay.mp4"
        frames_dir = output_dir / "frames"
        if config.save_intermediate_frames:
            frames_dir.mkdir(parents=True, exist_ok=True)

        model = self.registry.create(config.model)

        capture = cv2.VideoCapture(str(input_path))
        if not capture.isOpened():
            raise RuntimeError(f"Failed to open video: {input_path}")

        fps = capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30.0

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames > 0:
            duration = total_frames / fps
            if duration > self.max_video_seconds:
                raise ValueError(
                    f"Video too long ({duration:.2f}s). Max allowed: {self.max_video_seconds}s"
                )

        src_w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_w, out_h = self._resolve_output_resolution(config.output_resolution, src_w, src_h)

        writer = cv2.VideoWriter(
            str(overlay_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (out_w, out_h),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError("Failed to initialize output overlay video writer")

        frames: list[FramePoseRecord] = []
        frame_index = 0

        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                if (out_w, out_h) != (src_w, src_h):
                    frame = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

                pose_result: PoseResult
                if frame_index % config.every_n_frames == 0:
                    pose_result = model.infer(frame)
                else:
                    pose_result = PoseResult()

                frames.append(
                    FramePoseRecord(
                        frame_index=frame_index,
                        keypoints=pose_result.keypoints,
                        confidence=pose_result.confidence,
                        bbox=pose_result.bbox,
                    )
                )

                rendered = model.visualize(frame, pose_result)
                writer.write(rendered)

                if config.save_intermediate_frames:
                    cv2.imwrite(str(frames_dir / f"{frame_index:06d}.jpg"), rendered)

                frame_index += 1
        finally:
            capture.release()
            writer.release()

        payload = KeypointsPayload(
            video_id=video_id,
            model=config.model,
            frames=frames,
            meta={
                "fps": fps,
                "total_frames": frame_index,
                "output_resolution": f"{out_w}x{out_h}",
                "every_n_frames": config.every_n_frames,
            },
        )

        keypoints_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

        outputs = {
            "keypoints": str(keypoints_path),
            "overlay": str(overlay_path),
        }
        if config.save_intermediate_frames:
            outputs["frames_dir"] = str(frames_dir)
        return outputs

    @staticmethod
    def _resolve_output_resolution(output_resolution: str, src_w: int, src_h: int) -> tuple[int, int]:
        if output_resolution.lower() == "original":
            return src_w, src_h

        try:
            width_str, height_str = output_resolution.lower().split("x", maxsplit=1)
            width = int(width_str)
            height = int(height_str)
        except Exception as exc:  # pragma: no cover - defensive input validation
            raise ValueError("output_resolution must be 'original' or '<width>x<height>'") from exc

        if width <= 0 or height <= 0:
            raise ValueError("output_resolution width/height must be positive")
        return width, height

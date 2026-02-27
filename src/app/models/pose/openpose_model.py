from app.models.base_model import BasePoseModel, Frame
from app.models.pose.mediapipe_model import MediaPipePoseModel
from app.schemas.pose import PoseResult


class OpenPoseModel(BasePoseModel):
    """
    Adapter for an OpenPose-compatible interface.
    Current demo implementation delegates to MediaPipe as fallback.
    Replace this class internals with real OpenPose inference later.
    """

    name = "openpose"

    def __init__(self) -> None:
        self._fallback = MediaPipePoseModel()

    def load_model(self) -> None:
        self._fallback.load_model()

    def infer(self, frame: Frame) -> PoseResult:
        return self._fallback.infer(frame)

    def visualize(self, frame: Frame, pose_result: PoseResult) -> Frame:
        return self._fallback.visualize(frame, pose_result)

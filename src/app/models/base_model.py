from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from app.schemas.pose import PoseResult

Frame = npt.NDArray[np.uint8]


class BasePoseModel(ABC):
    name: str

    @abstractmethod
    def load_model(self) -> None:
        """Load model weights/resources."""

    @abstractmethod
    def infer(self, frame: Frame) -> PoseResult:
        """Run pose inference on a single frame."""

    @abstractmethod
    def visualize(self, frame: Frame, pose_result: PoseResult) -> Frame:
        """Draw pose result on frame and return a rendered frame."""

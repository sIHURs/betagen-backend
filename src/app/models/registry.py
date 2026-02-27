from collections.abc import Callable

from app.models.base_model import BasePoseModel
from app.models.pose.mediapipe_model import MediaPipePoseModel
from app.models.pose.openpose_model import OpenPoseModel


class PoseModelRegistry:
    def __init__(self) -> None:
        self._factories: dict[str, Callable[[], BasePoseModel]] = {
            "openpose": OpenPoseModel,
            "mediapipe": MediaPipePoseModel,
        }

    def register(self, name: str, factory: Callable[[], BasePoseModel]) -> None:
        self._factories[name] = factory

    def create(self, name: str) -> BasePoseModel:
        key = name.lower()
        if key not in self._factories:
            supported = ", ".join(sorted(self._factories))
            raise ValueError(f"Unsupported pose model '{name}'. Supported: {supported}")
        model = self._factories[key]()
        model.load_model()
        return model

    def supported_models(self) -> list[str]:
        return sorted(self._factories)

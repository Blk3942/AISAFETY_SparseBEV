# nuScenes dev-kit (subset).

import abc
from collections import defaultdict
from typing import List, Tuple, Union

import numpy as np


class EvalBox(abc.ABC):
    """Abstract base class for data classes used during detection evaluation."""

    def __init__(
        self,
        sample_token: str = "",
        translation: Tuple[float, float, float] = (0, 0, 0),
        size: Tuple[float, float, float] = (0, 0, 0),
        rotation: Tuple[float, float, float, float] = (0, 0, 0, 0),
        velocity: Tuple[float, float] = (0, 0),
        ego_translation: Tuple[float, float, float] = (0, 0, 0),
        num_pts: int = -1,
    ):
        assert type(sample_token) == str
        assert len(translation) == 3 and not np.any(np.isnan(translation))
        assert len(size) == 3 and not np.any(np.isnan(size))
        assert len(rotation) == 4 and not np.any(np.isnan(rotation))
        assert len(velocity) == 2  # velocity can include NaN in DB.
        assert len(ego_translation) == 3 and not np.any(np.isnan(ego_translation))
        assert type(num_pts) == int and not np.any(np.isnan(num_pts))

        self.sample_token = sample_token
        self.translation = translation
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.ego_translation = ego_translation
        self.num_pts = num_pts

    @property
    def ego_dist(self) -> float:
        return float(np.sqrt(np.sum(np.array(self.ego_translation[:2]) ** 2)))

    def __repr__(self):
        return str(self.serialize())

    @abc.abstractmethod
    def serialize(self) -> dict:
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        raise NotImplementedError


EvalBoxType = Union['DetectionBox']


class EvalBoxes:
    """Groups EvalBox instances by sample."""

    def __init__(self):
        self.boxes = defaultdict(list)

    def __repr__(self):
        return f"EvalBoxes with {len(self.all)} boxes across {len(self.sample_tokens)} samples"

    def __getitem__(self, item) -> List[EvalBoxType]:
        return self.boxes[item]

    def __len__(self):
        return len(self.boxes)

    @property
    def all(self) -> List[EvalBoxType]:
        ab = []
        for sample_token in self.sample_tokens:
            ab.extend(self[sample_token])
        return ab

    @property
    def sample_tokens(self) -> List[str]:
        return list(self.boxes.keys())

    def add_boxes(self, sample_token: str, boxes: List[EvalBoxType]) -> None:
        self.boxes[sample_token].extend(boxes)

    def serialize(self) -> dict:
        return {key: [box.serialize() for box in boxes] for key, boxes in self.boxes.items()}

    @classmethod
    def deserialize(cls, content: dict, box_cls):
        eb = cls()
        for sample_token, boxes in content.items():
            eb.add_boxes(sample_token, [box_cls.deserialize(box) for box in boxes])
        return eb


class MetricData(abc.ABC):
    @abc.abstractmethod
    def serialize(self):
        raise NotImplementedError

    @classmethod
    @abc.abstractmethod
    def deserialize(cls, content: dict):
        raise NotImplementedError


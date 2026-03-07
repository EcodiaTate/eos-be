from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field

from primitives.common import EOSBaseModel


class InteractiveImitationSignal(EOSBaseModel):
    """
    Signal representing an interactive imitation learning event from a phone interface.
    Follows RoboPocket paper (2603.05504) interaction model.
    """
    device_id: str = Field(..., description="Unique identifier for the phone/device")
    action_type: Literal["demonstration", "correction", "feedback"] = Field(
        ..., description="Type of interactive imitation signal"
    )
    robot_policy_id: str = Field(..., description="Identifier of the robot policy being learned")
    confidence_score: float = Field(
        ge=0.0, le=1.0, description="Confidence of the imitation signal (0.0-1.0)"
    )
    raw_data: bytes | None = Field(
        default=None, description="Optional raw sensor/video data for advanced learning"
    )
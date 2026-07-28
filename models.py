from pydantic import BaseModel
from typing import List, Dict, Any

class BowlRequest(BaseModel):
    speed_kmh: float = 132.0
    length_m: float = 7.4
    line_m: float = 0.1
    swing_cm: float = 8.0
    seam_cm: float = 5.0
    shot_type: str = "drive"

class TrajectoryPoint(BaseModel):
    x: float
    y: float
    z: float

class DeliveryResult(BaseModel):
    id: str
    trajectory: List[TrajectoryPoint]
    bounce_x: float
    bounce_y: float
    release_speed_kmh: float
    length_category: str
    line_category: str
    swing_cm: float
    seam_cm: float
    spin_rpm: float
    valid_event: bool
    fusion_confidence: float
    shot_type: str
    contact_quality: str
    timing: str
    footwork: str
    analytics: Dict[str, Any]

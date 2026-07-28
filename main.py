from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from models import BowlRequest, DeliveryResult, TrajectoryPoint
import numpy as np
import uuid
import random
from typing import List

app = FastAPI(title="Multi-Modal Edge-AI Cricket Backend", version="4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

session_deliveries: List[DeliveryResult] = []

def generate_trajectory(speed_kmh, length_m, line_m, swing_cm, seam_cm):
    v0 = speed_kmh / 3.6
    swing = swing_cm / 100.0
    seam = seam_cm / 100.0
    release_height = 2.15
    t_bounce = max(0.37, min(0.95, length_m / max(v0, 1) * 1.06))

    points = []
    steps = 90

    for i in range(steps + 1):
        t = (i / steps) * t_bounce
        x = v0 * t * 0.975
        progress = t / t_bounce
        y = line_m + swing * (progress ** 1.45)
        z = release_height - 0.5 * 9.81 * t * t * 0.91
        points.append({"x": float(x), "y": float(y), "z": float(max(z, 0.04))})

    bounce_x = points[-1]["x"]
    bounce_y = points[-1]["y"]

    v_post = v0 * 0.64
    for i in range(1, 50):
        t = (i / 49) * 0.58
        x = bounce_x + v_post * t
        y = bounce_y + seam * (t / 0.58)
        z = 0.11 + v_post * 0.21 * t - 0.5 * 9.81 * t * t * 0.76
        points.append({"x": float(x), "y": float(y), "z": float(max(z, 0.04))})

    return points, bounce_x, bounce_y

def stage_a_time_sync(piezo_amplitude: float):
    return {
        "master_trigger": True,
        "sync_window_ms": 50,
        "piezo_amplitude": piezo_amplitude
    }

def stage_b_cross_validation(piezo_amp, has_radar, has_nir):
    valid = piezo_amp > 0.28 and has_radar and has_nir
    return {
        "valid_event": valid,
        "false_positive_rejected": not valid
    }

def stage_c_high_level_ml(length_m, line_m, shot_type, valid):
    if length_m < 4.5:
        length_cat = "Yorker / Full"
    elif length_m < 7.5:
        length_cat = "Good Length"
    elif length_m < 10.5:
        length_cat = "Short of Length"
    else:
        length_cat = "Bouncer / Short"

    if abs(line_m) < 0.18:
        line_cat = "Off Stump"
    elif line_m < -0.35:
        line_cat = "Wide Outside Off"
    elif line_m > 0.35:
        line_cat = "Leg Side"
    else:
        line_cat = "Middle / Leg"

    contact = random.choice(["Sweet Spot", "Near Sweet Spot", "Edge", "Toe", "Missed"]) if shot_type != "leave" else "Left"
    timing = random.choice(["Early", "On Time", "Late"]) if shot_type != "leave" else "N/A"
    footwork = random.choice(["Front Foot", "Back Foot", "Neutral", "No Movement"])

    return {
        "length_category": length_cat,
        "line_category": line_cat,
        "shot_type": shot_type,
        "contact_quality": contact,
        "timing": timing,
        "footwork": footwork,
        "fusion_confidence": 0.94 if valid else 0.25
    }

@app.post("/api/bowl", response_model=DeliveryResult)
def bowl_delivery(req: BowlRequest):
    points, bounce_x, bounce_y = generate_trajectory(
        req.speed_kmh, req.length_m, req.line_m, req.swing_cm, req.seam_cm
    )

    piezo_amp = min(1.0, req.speed_kmh / 145.0)
    validation = stage_b_cross_validation(piezo_amp, True, True)
    ml = stage_c_high_level_ml(req.length_m, req.line_m, req.shot_type, validation["valid_event"])
    sync = stage_a_time_sync(piezo_amp)

    result = DeliveryResult(
        id=str(uuid.uuid4())[:8],
        trajectory=[TrajectoryPoint(**p) for p in points],
        bounce_x=bounce_x,
        bounce_y=bounce_y,
        release_speed_kmh=req.speed_kmh,
        length_category=ml["length_category"],
        line_category=ml["line_category"],
        swing_cm=req.swing_cm,
        seam_cm=req.seam_cm,
        spin_rpm=random.randint(0, 2800),
        valid_event=validation["valid_event"],
        fusion_confidence=ml["fusion_confidence"],
        shot_type=ml["shot_type"],
        contact_quality=ml["contact_quality"],
        timing=ml["timing"],
        footwork=ml["footwork"],
        analytics={
            "stage_a": sync,
            "stage_b": validation,
            "stage_c": ml
        }
    )

    session_deliveries.append(result)
    return result

@app.get("/api/session")
def get_session():
    return {
        "count": len(session_deliveries),
        "deliveries": session_deliveries
    }

@app.post("/api/session/clear")
def clear_session():
    session_deliveries.clear()
    return {"message": "Session cleared"}

@app.get("/api/health")
def health():
    return {"status": "ok", "system": "Multi-Modal Edge-AI Cricket v4.0"}

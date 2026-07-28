# multi_modal_cricket_sim.py
# Multi-Modal Edge-AI Sensor Fusion Simulator for Indoor Cricket (v4.0 Spec)
# Run with: streamlit run multi_modal_cricket_sim.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import io
import base64

st.set_page_config(
    page_title="Multi-Modal Edge-AI Cricket Simulator",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# 1. SENSOR MODELS (matching Section 2 of the specification)
# =============================================================================

@dataclass
class PiezoEvent:
    timestamp_us: int
    location: str          # "pitch", "stumps", "frame"
    amplitude: float       # normalized 0-1
    frequency_hz: float

@dataclass
class NIREvent:
    timestamp_us: int
    zone: str              # "release", "popping_crease", "hitting"
    entry_speed_ms: float
    exit_speed_ms: float
    gate_x: float
    gate_y: float

@dataclass
class UltrasonicReading:
    timestamp_us: int
    distance_cm: float
    zone: str              # "stance", "side", "overhead"
    sway_cm: float

@dataclass
class RadarTrack:
    timestamps_us: np.ndarray
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    velocity_ms: np.ndarray
    spin_rpm: float
    micro_doppler: np.ndarray

@dataclass
class SpatialAIFrame:
    timestamp_us: int
    skeleton_3d: Dict[str, Tuple[float, float, float]]  # joint -> (x,y,z)
    bat_face_normal: Tuple[float, float, float]
    depth_confidence: float
    occlusion_flag: bool

# =============================================================================
# 2. PHYSICS & EVENT GENERATORS
# =============================================================================

def generate_ball_trajectory(
    release_speed_kmh: float,
    length_m: float,          # 0 = full, ~6-8 = good length, >10 = short
    line_offset_m: float,     # negative = off, positive = leg (from bowler view)
    swing_m: float = 0.0,     # lateral deviation before bounce
    seam_m: float = 0.0,      # post-bounce deviation
    bounce_factor: float = 0.65,
    spin_rpm: float = 0.0
) -> Dict:
    """Generate a realistic 3D ball trajectory for indoor nets (~20 m pitch)."""
    pitch_length = 20.12  # standard indoor net length approximation
    release_height = 2.1
    release_x = 0.0
    release_y = 0.0

    # Convert speed
    v0 = release_speed_kmh / 3.6  # m/s

    # Time of flight to bounce
    # Simple parabolic + empirical bounce model
    t_bounce = np.sqrt(2 * (release_height - 0.15) / 9.81) * 0.85  # empirical
    t_bounce = max(0.4, min(0.9, length_m / v0 * 1.1))

    # Pre-bounce path
    t1 = np.linspace(0, t_bounce, 40)
    x1 = release_x + v0 * t1 * 0.98
    y1 = line_offset_m + swing_m * (t1 / t_bounce)**1.5
    z1 = release_height - 0.5 * 9.81 * t1**2 * 0.9

    # Bounce point
    bounce_x = x1[-1]
    bounce_y = y1[-1]
    bounce_z = 0.12

    # Post-bounce
    v_post = v0 * bounce_factor
    t2 = np.linspace(0, 0.6, 30)
    x2 = bounce_x + v_post * t2
    y2 = bounce_y + seam_m * (t2 / 0.6)
    z2 = bounce_z + v_post * 0.25 * t2 - 0.5 * 9.81 * t2**2 * 0.7
    z2 = np.maximum(z2, 0.05)

    # Combine
    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    z = np.concatenate([z1, z2[1:]])
    t = np.concatenate([t1, t_bounce + t2[1:]])

    # Velocity profile
    vx = np.gradient(x, t)
    speed = np.sqrt(vx**2 + np.gradient(y, t)**2 + np.gradient(z, t)**2)

    return {
        "t": t,
        "x": x, "y": y, "z": z,
        "speed_ms": speed,
        "release_speed_kmh": release_speed_kmh,
        "bounce_x": bounce_x,
        "bounce_y": bounce_y,
        "length_m": bounce_x,
        "line_m": bounce_y,
        "spin_rpm": spin_rpm,
        "swing_m": swing_m,
        "seam_m": seam_m
    }

def simulate_sensors(traj: Dict, shot_type: str = "leave") -> Dict:
    """Generate synthetic multi-modal sensor events from a trajectory + shot."""
    t0_us = int(datetime.now().timestamp() * 1e6)

    # --- A. Piezoelectric ---
    piezo_events = []
    # Impact at bounce
    bounce_idx = np.argmin(np.abs(traj["z"] - 0.12))
    piezo_events.append(PiezoEvent(
        timestamp_us=t0_us + int(traj["t"][bounce_idx] * 1e6),
        location="pitch",
        amplitude=min(1.0, traj["speed_ms"][bounce_idx] / 35),
        frequency_hz=random.uniform(80, 180)
    ))
    # Bat impact if not leave
    if shot_type != "leave":
        impact_t = traj["t"][bounce_idx] + random.uniform(0.08, 0.18)
        piezo_events.append(PiezoEvent(
            timestamp_us=t0_us + int(impact_t * 1e6),
            location="bat",
            amplitude=random.uniform(0.6, 1.0),
            frequency_hz=random.uniform(120, 250)
        ))

    # --- B. NIR Light Curtains ---
    nir_events = []
    # Release gate
    nir_events.append(NIREvent(
        timestamp_us=t0_us,
        zone="release",
        entry_speed_ms=traj["speed_ms"][0],
        exit_speed_ms=traj["speed_ms"][5],
        gate_x=0.0, gate_y=traj["y"][0]
    ))
    # Popping crease
    crease_idx = np.argmin(np.abs(traj["x"] - 17.7))
    nir_events.append(NIREvent(
        timestamp_us=t0_us + int(traj["t"][crease_idx] * 1e6),
        zone="popping_crease",
        entry_speed_ms=traj["speed_ms"][crease_idx],
        exit_speed_ms=traj["speed_ms"][min(crease_idx+3, len(traj["speed_ms"])-1)],
        gate_x=traj["x"][crease_idx], gate_y=traj["y"][crease_idx]
    ))

    # --- C. Ultrasonic (stance) ---
    ultra = [
        UltrasonicReading(t0_us + i*20000, 45 + random.uniform(-8, 8), "stance", random.uniform(-3, 3))
        for i in range(15)
    ]

    # --- D. mmWave Radar Track ---
    radar = RadarTrack(
        timestamps_us=t0_us + (traj["t"] * 1e6).astype(int),
        x=traj["x"], y=traj["y"], z=traj["z"],
        velocity_ms=traj["speed_ms"],
        spin_rpm=traj["spin_rpm"],
        micro_doppler=np.sin(np.linspace(0, 8*np.pi, len(traj["t"]))) * traj["spin_rpm"]/300
    )

    # --- E. Spatial AI (simplified skeleton + bat) ---
    # Dummy skeleton at impact
    impact_t_us = t0_us + int((traj["t"][bounce_idx] + 0.12) * 1e6)
    skeleton = {
        "head": (18.2, 0.1, 1.75),
        "shoulder_l": (18.1, -0.25, 1.45),
        "shoulder_r": (18.1, 0.25, 1.45),
        "elbow_r": (18.0, 0.45, 1.15),
        "wrist_r": (17.9, 0.55, 0.95),
        "hip": (18.3, 0.0, 0.95),
        "knee_l": (18.4, -0.15, 0.5),
        "knee_r": (18.4, 0.15, 0.5),
        "ankle_l": (18.5, -0.15, 0.05),
        "ankle_r": (18.5, 0.15, 0.05),
    }
    spatial = SpatialAIFrame(
        timestamp_us=impact_t_us,
        skeleton_3d=skeleton,
        bat_face_normal=(0.2, 0.1, 0.97),
        depth_confidence=0.92,
        occlusion_flag=False
    )

    return {
        "piezo": piezo_events,
        "nir": nir_events,
        "ultrasonic": ultra,
        "radar": radar,
        "spatial": spatial,
        "traj": traj,
        "shot_type": shot_type
    }

# =============================================================================
# 3. EDGE-AI FUSION PIPELINE (Section 4)
# =============================================================================

def stage_a_time_sync(sensor_data: Dict) -> Dict:
    """Stage A: Low-Level Time Synchronization using Piezo as master trigger."""
    if not sensor_data["piezo"]:
        return sensor_data
    master_ts = sensor_data["piezo"][0].timestamp_us
    # Align everything to a ±25 ms window around master
    window = 50_000  # 50 ms
    sensor_data["sync_window_us"] = (master_ts - window//2, master_ts + window//2)
    sensor_data["master_trigger"] = master_ts
    return sensor_data

def stage_b_cross_validation(sensor_data: Dict) -> Dict:
    """Stage B: Cross-validate piezo vs radar vs NIR to kill false positives."""
    piezo_amps = [p.amplitude for p in sensor_data["piezo"]]
    has_radar_track = len(sensor_data["radar"].x) > 10
    has_nir = len(sensor_data["nir"]) >= 1

    valid_impact = any(a > 0.3 for a in piezo_amps) and has_radar_track and has_nir
    sensor_data["valid_event"] = valid_impact
    sensor_data["false_positive_rejected"] = not valid_impact
    return sensor_data

def stage_c_high_level_ml(sensor_data: Dict) -> Dict:
    """Stage C: Lightweight classification of delivery & shot."""
    traj = sensor_data["traj"]
    shot = sensor_data["shot_type"]

    # Delivery classification
    length = traj["length_m"]
    if length < 4.5:
        length_cat = "Full / Yorker"
    elif length < 7.5:
        length_cat = "Good Length"
    elif length < 10:
        length_cat = "Short of Length"
    else:
        length_cat = "Bouncer / Short"

    line = traj["line_m"]
    if abs(line) < 0.15:
        line_cat = "Off Stump"
    elif line < -0.3:
        line_cat = "Wide Outside Off"
    elif line > 0.3:
        line_cat = "Leg Side"
    else:
        line_cat = "Middle / Leg"

    # Shot quality (mock)
    contact_quality = random.choice(["Sweet Spot", "Near Sweet Spot", "Edge", "Toe", "Missed"]) if shot != "leave" else "Left"
    timing = random.choice(["Early", "On Time", "Late"]) if shot != "leave" else "N/A"

    sensor_data["analytics"] = {
        "delivery": {
            "length_category": length_cat,
            "line_category": line_cat,
            "release_speed_kmh": round(traj["release_speed_kmh"], 1),
            "speed_at_bounce_kmh": round(traj["speed_ms"][np.argmin(np.abs(traj["z"]-0.12))] * 3.6, 1),
            "swing_cm": round(traj["swing_m"] * 100, 1),
            "seam_cm": round(traj["seam_m"] * 100, 1),
            "spin_rpm": traj["spin_rpm"]
        },
        "batting": {
            "shot_type": shot,
            "contact_quality": contact_quality,
            "timing": timing,
            "footwork": random.choice(["Front Foot", "Back Foot", "Neutral", "No Movement"])
        },
        "fusion_confidence": 0.94 if sensor_data["valid_event"] else 0.2
    }
    return sensor_data

def run_fusion_pipeline(sensor_data: Dict) -> Dict:
    data = stage_a_time_sync(sensor_data)
    data = stage_b_cross_validation(data)
    data = stage_c_high_level_ml(data)
    return data

# =============================================================================
# 4. VISUALIZATION HELPERS
# =============================================================================

def plot_3d_trajectory(traj: Dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=traj["x"], y=traj["y"], z=traj["z"],
        mode="lines+markers",
        marker=dict(size=3, color=traj["speed_ms"], colorscale="Viridis", showscale=True, colorbar=dict(title="Speed m/s")),
        line=dict(width=6, color="orange"),
        name="Ball Path"
    ))
    # Pitch surface
    fig.add_trace(go.Mesh3d(
        x=[0, 20, 20, 0], y=[-1.5, -1.5, 1.5, 1.5], z=[0, 0, 0, 0],
        opacity=0.3, color="green", name="Pitch"
    ))
    fig.update_layout(
        title="3D Ball Trajectory (mmWave + Spatial AI)",
        scene=dict(
            xaxis_title="Length (m)",
            yaxis_title="Line (m)",
            zaxis_title="Height (m)",
            aspectmode="manual",
            aspectratio=dict(x=2, y=1, z=0.4)
        ),
        height=500
    )
    return fig

def plot_pitch_map(events: List[Dict]):
    fig = go.Figure()
    # Pitch outline
    fig.add_shape(type="rect", x0=0, y0=-1.3, x1=20.12, y1=1.3, line=dict(color="white"), fillcolor="rgba(34,139,34,0.3)")
    # Stumps
    for sx in [0, 20.12]:
        fig.add_shape(type="line", x0=sx, y0=-0.11, x1=sx, y1=0.11, line=dict(color="yellow", width=4))

    xs, ys, speeds, texts = [], [], [], []
    for e in events:
        t = e["traj"]
        xs.append(t["bounce_x"])
        ys.append(t["bounce_y"])
        speeds.append(t["release_speed_kmh"])
        texts.append(f"{t['release_speed_kmh']:.0f} km/h<br>{e['analytics']['delivery']['length_category']}")

    fig.add_trace(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=14, color=speeds, colorscale="Hot", showscale=True, colorbar=dict(title="km/h"),
                    line=dict(width=1, color="black")),
        text=texts, hoverinfo="text",
        name="Pitch Map"
    ))
    fig.update_layout(
        title="Pitch Map (Line & Length) – Piezo + Radar + NIR",
        xaxis_title="Length (m from bowler)",
        yaxis_title="Line (m, − = off)",
        yaxis=dict(scaleanchor="x", scaleratio=1, range=[-1.5, 1.5]),
        xaxis=dict(range=[-0.5, 21]),
        height=450,
        plot_bgcolor="rgba(20,80,20,0.2)"
    )
    return fig

def plot_speed_profile(traj: Dict):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=traj["t"]*1000, y=traj["speed_ms"]*3.6,
        mode="lines", line=dict(width=3, color="#e74c3c"),
        name="Ball Speed"
    ))
    fig.update_layout(
        title="Ball Speed Profile (Release → Bounce → Bat)",
        xaxis_title="Time (ms)",
        yaxis_title="Speed (km/h)",
        height=350
    )
    return fig

# =============================================================================
# 5. STREAMLIT UI
# =============================================================================

st.title("🏏 Multi-Modal Edge-AI Cricket Tracking Simulator")
st.caption("Technical Specification v4.0 — Indoor Net Sensor Fusion | Zero-Cloud | Edge-AI Pipeline")

with st.sidebar:
    st.header("Session Controls")
    n_balls = st.slider("Number of deliveries", 1, 24, 6)
    base_speed = st.slider("Base release speed (km/h)", 70, 150, 125)
    spin_type = st.selectbox("Bowling type", ["Pace", "Medium", "Off-Spin", "Leg-Spin", "Wrist-Spin"])
    st.markdown("---")
    st.subheader("Sensor Status")
    st.success("Piezoelectric  ✓ Active")
    st.success("NIR Light Curtains  ✓ Active")
    st.success("Ultrasonic Array  ✓ Active")
    st.success("mmWave Radar  ✓ Active")
    st.success("6× OAK-D Spatial AI  ✓ Active")
    st.markdown("---")
    if st.button("🔄 Run Full Fusion Pipeline", type="primary", use_container_width=True):
        st.session_state.run_sim = True

if "run_sim" not in st.session_state:
    st.session_state.run_sim = False
if "session_events" not in st.session_state:
    st.session_state.session_events = []

if st.session_state.run_sim:
    st.session_state.session_events = []
    progress = st.progress(0, text="Generating multi-modal sensor streams…")

    for i in range(n_balls):
        # Randomize delivery parameters
        speed = base_speed + random.uniform(-12, 12)
        length = random.uniform(3.5, 12.5)
        line = random.uniform(-0.7, 0.6)
        swing = random.uniform(-0.25, 0.25) if spin_type == "Pace" else random.uniform(-0.08, 0.08)
        seam = random.uniform(-0.18, 0.18)
        spin = {"Pace": 0, "Medium": 300, "Off-Spin": 1800, "Leg-Spin": 2200, "Wrist-Spin": 2800}[spin_type]
        spin += random.uniform(-200, 200)

        shot = random.choices(
            ["drive", "pull", "cut", "sweep", "leave", "defend", "hook"],
            weights=[0.2, 0.12, 0.15, 0.08, 0.25, 0.15, 0.05]
        )[0]

        traj = generate_ball_trajectory(speed, length, line, swing, seam, spin_rpm=spin)
        sensors = simulate_sensors(traj, shot)
        fused = run_fusion_pipeline(sensors)
        st.session_state.session_events.append(fused)
        progress.progress((i+1)/n_balls, text=f"Fused delivery {i+1}/{n_balls}")

    st.session_state.run_sim = False
    progress.empty()
    st.success(f"Pipeline complete — {n_balls} deliveries fused with <20 ms edge latency (simulated)")

# ---- Results Dashboard ----
events = st.session_state.session_events
if events:
    st.header("Integrated Analytics Suite")

    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    avg_speed = np.mean([e["traj"]["release_speed_kmh"] for e in events])
    valid = sum(1 for e in events if e["valid_event"])
    c1.metric("Avg Release Speed", f"{avg_speed:.1f} km/h")
    c2.metric("Valid Events (Stage B)", f"{valid}/{len(events)}")
    c3.metric("False Positives Rejected", f"{len(events)-valid}")
    c4.metric("Avg Fusion Confidence", f"{np.mean([e['analytics']['fusion_confidence'] for e in events])*100:.0f}%")
    c5.metric("Edge Latency (sim)", "< 18 ms")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "3D Trajectory & Speed", "Pitch Map & Beehive", "Bowling Analytics", "Batting Analytics", "Fusion Pipeline Trace"
    ])

    with tab1:
        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_3d_trajectory(events[0]["traj"]), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_speed_profile(events[0]["traj"]), use_container_width=True)
        st.caption("Ball 1 shown. mmWave continuous track + Piezo master timestamp + Spatial AI depth.")

    with tab2:
        st.plotly_chart(plot_pitch_map(events), use_container_width=True)
        st.caption("Every bounce location from fused Piezo + Radar + NIR. Color = release speed.")

    with tab3:
        st.subheader("Bowling KPIs (per delivery)")
        rows = []
        for i, e in enumerate(events):
            d = e["analytics"]["delivery"]
            rows.append({
                "Ball": i+1,
                "Speed (km/h)": d["release_speed_kmh"],
                "Length": d["length_category"],
                "Line": d["line_category"],
                "Swing (cm)": d["swing_cm"],
                "Seam (cm)": d["seam_cm"],
                "Spin (rpm)": d["spin_rpm"],
                "Speed @ Bounce": d["speed_at_bounce_kmh"]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Simple consistency chart
        speeds = [e["traj"]["release_speed_kmh"] for e in events]
        fig = px.line(x=list(range(1, len(speeds)+1)), y=speeds, markers=True,
                      labels={"x": "Ball Number", "y": "Release Speed (km/h)"},
                      title="Release Speed Consistency Across Spell")
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Batting Analytics (fused Spatial AI + Piezo impact + Radar)")
        bat_rows = []
        for i, e in enumerate(events):
            b = e["analytics"]["batting"]
            bat_rows.append({
                "Ball": i+1,
                "Shot Type": b["shot_type"].title(),
                "Contact Quality": b["contact_quality"],
                "Timing": b["timing"],
                "Footwork": b["footwork"],
                "Fusion Conf.": f"{e['analytics']['fusion_confidence']*100:.0f}%"
            })
        st.dataframe(pd.DataFrame(bat_rows), use_container_width=True, hide_index=True)

        # Shot distribution
        shot_counts = pd.Series([e["analytics"]["batting"]["shot_type"] for e in events]).value_counts()
        fig = px.pie(values=shot_counts.values, names=shot_counts.index, title="Shot Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        st.subheader("Edge-AI Pipeline Trace (Section 4)")
        e = events[0]
        st.markdown(f"""
        **Stage A – Time Synchronization**  
        Master Piezo trigger: `{e.get('master_trigger', 'N/A')}` µs  
        Sync window: ±25 ms around impact  

        **Stage B – Cross-Validation**  
        Valid event: `{e['valid_event']}`  
        False-positive rejected: `{e['false_positive_rejected']}`  
        (Piezo amplitude + Radar track + NIR gate must all agree)

        **Stage C – High-Level ML**  
        Delivery classified as **{e['analytics']['delivery']['length_category']} / {e['analytics']['delivery']['line_category']}**  
        Shot: **{e['analytics']['batting']['shot_type']}** | Contact: **{e['analytics']['batting']['contact_quality']}**  
        Overall fusion confidence: **{e['analytics']['fusion_confidence']*100:.0f}%**
        """)
        st.info("All processing occurs on the local Edge-AI box. No data leaves the facility (zero cloud latency).")

else:
    st.info("Configure the session in the sidebar and click **Run Full Fusion Pipeline** to generate a multi-modal simulation.")

st.markdown("---")
st.caption("Simulation of the Multi-Modal Edge-AI Sensor Fusion Architecture (Technical Spec v4.0). "
           "Piezo + NIR + Ultrasonic + mmWave + 6× OAK-D → three-stage Edge pipeline → coach analytics.")

# multi_modal_cricket_sim.py
# Multi-Modal Edge-AI Cricket Simulator + Animated Ball Flight (Option A)
# Deploy-ready for Render / Streamlit

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import os

# Production settings for Render
os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

st.set_page_config(
    page_title="Multi-Modal Edge-AI Cricket Simulator",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# SENSOR & PHYSICS (same as before)
# =============================================================================

@dataclass
class PiezoEvent:
    timestamp_us: int
    location: str
    amplitude: float
    frequency_hz: float

@dataclass
class NIREvent:
    timestamp_us: int
    zone: str
    entry_speed_ms: float
    exit_speed_ms: float
    gate_x: float
    gate_y: float

@dataclass
class UltrasonicReading:
    timestamp_us: int
    distance_cm: float
    zone: str
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
    skeleton_3d: Dict[str, Tuple[float, float, float]]
    bat_face_normal: Tuple[float, float, float]
    depth_confidence: float
    occlusion_flag: bool


def generate_ball_trajectory(
    release_speed_kmh: float,
    length_m: float,
    line_offset_m: float,
    swing_m: float = 0.0,
    seam_m: float = 0.0,
    bounce_factor: float = 0.65,
    spin_rpm: float = 0.0
) -> Dict:
    release_height = 2.1
    v0 = release_speed_kmh / 3.6

    t_bounce = max(0.38, min(0.95, length_m / max(v0, 1) * 1.05))
    t1 = np.linspace(0, t_bounce, 50)
    x1 = v0 * t1 * 0.98
    y1 = line_offset_m + swing_m * (t1 / t_bounce)**1.4
    z1 = release_height - 0.5 * 9.81 * t1**2 * 0.92

    bounce_x, bounce_y = x1[-1], y1[-1]
    v_post = v0 * bounce_factor
    t2 = np.linspace(0, 0.55, 35)
    x2 = bounce_x + v_post * t2
    y2 = bounce_y + seam_m * (t2 / 0.55)
    z2 = 0.12 + v_post * 0.22 * t2 - 0.5 * 9.81 * t2**2 * 0.75
    z2 = np.maximum(z2, 0.04)

    x = np.concatenate([x1, x2[1:]])
    y = np.concatenate([y1, y2[1:]])
    z = np.concatenate([z1, z2[1:]])
    t = np.concatenate([t1, t_bounce + t2[1:]])
    speed = np.sqrt(np.gradient(x, t)**2 + np.gradient(y, t)**2 + np.gradient(z, t)**2)

    return {
        "t": t, "x": x, "y": y, "z": z, "speed_ms": speed,
        "release_speed_kmh": release_speed_kmh,
        "bounce_x": bounce_x, "bounce_y": bounce_y,
        "length_m": bounce_x, "line_m": bounce_y,
        "spin_rpm": spin_rpm, "swing_m": swing_m, "seam_m": seam_m,
        "t_bounce": t_bounce
    }


def simulate_sensors(traj: Dict, shot_type: str = "leave") -> Dict:
    t0_us = int(datetime.now().timestamp() * 1e6)
    bounce_idx = np.argmin(np.abs(traj["z"] - 0.12))

    piezo = [PiezoEvent(t0_us + int(traj["t"][bounce_idx]*1e6), "pitch",
                        min(1.0, traj["speed_ms"][bounce_idx]/35), random.uniform(80, 180))]
    if shot_type != "leave":
        piezo.append(PiezoEvent(t0_us + int((traj["t"][bounce_idx]+0.12)*1e6), "bat",
                                random.uniform(0.65, 1.0), random.uniform(130, 240)))

    nir = [
        NIREvent(t0_us, "release", traj["speed_ms"][0], traj["speed_ms"][4], 0.0, traj["y"][0]),
        NIREvent(t0_us + int(traj["t"][min(len(traj["t"])-1, 40)]*1e6), "popping_crease",
                 traj["speed_ms"][30], traj["speed_ms"][33], traj["x"][30], traj["y"][30])
    ]

    ultra = [UltrasonicReading(t0_us + i*20000, 45 + random.uniform(-7, 7), "stance", random.uniform(-3, 3))
             for i in range(12)]

    radar = RadarTrack(
        t0_us + (traj["t"]*1e6).astype(int),
        traj["x"], traj["y"], traj["z"], traj["speed_ms"],
        traj["spin_rpm"],
        np.sin(np.linspace(0, 7*np.pi, len(traj["t"]))) * traj["spin_rpm"]/350
    )

    skeleton = {
        "head": (18.3, 0.05, 1.72), "shoulder_r": (18.15, 0.28, 1.42),
        "elbow_r": (18.0, 0.48, 1.12), "wrist_r": (17.85, 0.58, 0.92),
        "hip": (18.4, 0.0, 0.92), "ankle_r": (18.55, 0.12, 0.05)
    }
    spatial = SpatialAIFrame(t0_us + int((traj["t"][bounce_idx]+0.11)*1e6),
                             skeleton, (0.15, 0.08, 0.98), 0.93, False)

    return {"piezo": piezo, "nir": nir, "ultrasonic": ultra, "radar": radar,
            "spatial": spatial, "traj": traj, "shot_type": shot_type}


def stage_a_time_sync(d: Dict) -> Dict:
    if d["piezo"]:
        master = d["piezo"][0].timestamp_us
        d["master_trigger"] = master
        d["sync_window_us"] = (master - 25000, master + 25000)
    return d

def stage_b_cross_validation(d: Dict) -> Dict:
    valid = (any(p.amplitude > 0.28 for p in d["piezo"]) and
             len(d["radar"].x) > 8 and len(d["nir"]) >= 1)
    d["valid_event"] = valid
    d["false_positive_rejected"] = not valid
    return d

def stage_c_high_level_ml(d: Dict) -> Dict:
    traj = d["traj"]
    length = traj["length_m"]
    length_cat = ("Full / Yorker" if length < 4.5 else
                  "Good Length" if length < 7.5 else
                  "Short of Length" if length < 10 else "Bouncer / Short")
    line = traj["line_m"]
    line_cat = ("Off Stump" if abs(line) < 0.18 else
                "Wide Outside Off" if line < -0.35 else
                "Leg Side" if line > 0.35 else "Middle / Leg")

    shot = d["shot_type"]
    contact = random.choice(["Sweet Spot", "Near Sweet Spot", "Edge", "Toe", "Missed"]) if shot != "leave" else "Left"
    timing = random.choice(["Early", "On Time", "Late"]) if shot != "leave" else "N/A"

    d["analytics"] = {
        "delivery": {
            "length_category": length_cat, "line_category": line_cat,
            "release_speed_kmh": round(traj["release_speed_kmh"], 1),
            "speed_at_bounce_kmh": round(traj["speed_ms"][np.argmin(np.abs(traj["z"]-0.12))]*3.6, 1),
            "swing_cm": round(traj["swing_m"]*100, 1),
            "seam_cm": round(traj["seam_m"]*100, 1),
            "spin_rpm": int(traj["spin_rpm"])
        },
        "batting": {
            "shot_type": shot, "contact_quality": contact,
            "timing": timing,
            "footwork": random.choice(["Front Foot", "Back Foot", "Neutral", "No Movement"])
        },
        "fusion_confidence": 0.94 if d["valid_event"] else 0.22
    }
    return d

def run_fusion_pipeline(d: Dict) -> Dict:
    return stage_c_high_level_ml(stage_b_cross_validation(stage_a_time_sync(d)))


# =============================================================================
# ANIMATED REPLAYS (Option A)
# =============================================================================

def create_side_on_animation(traj: Dict, speed_multiplier: float = 1.0):
    """Side-on view: ball travels left→right, bounce clearly visible."""
    t = traj["t"]
    x = traj["x"]
    z = traj["z"]
    n_frames = len(t)

    # Subsample for smoother animation if needed
    step = max(1, n_frames // 60)
    frames_idx = list(range(0, n_frames, step))
    if frames_idx[-1] != n_frames - 1:
        frames_idx.append(n_frames - 1)

    fig = go.Figure()

    # Pitch surface line
    fig.add_trace(go.Scatter(
        x=[0, 20.5], y=[0, 0],
        mode="lines", line=dict(color="forestgreen", width=8),
        name="Pitch", showlegend=False
    ))

    # Stumps (simple)
    fig.add_trace(go.Scatter(
        x=[20.12, 20.12], y=[0, 0.71],
        mode="lines", line=dict(color="yellow", width=6),
        name="Stumps", showlegend=False
    ))

    # Initial empty ball + trail
    fig.add_trace(go.Scatter(
        x=[x[0]], y=[z[0]],
        mode="markers",
        marker=dict(size=14, color="red", line=dict(width=2, color="white")),
        name="Ball"
    ))
    fig.add_trace(go.Scatter(
        x=[], y=[],
        mode="lines",
        line=dict(width=3, color="orange"),
        name="Trail", showlegend=False
    ))

    frames = []
    for i in frames_idx:
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[x[i]], y=[z[i]]),                    # ball
                go.Scatter(x=x[:i+1], y=z[:i+1])                   # trail
            ],
            name=str(i),
            traces=[2, 3]
        ))

    fig.frames = frames

    # Play button + slider
    fig.update_layout(
        title="Side-On Ball Flight (Release → Bounce → Bat)",
        xaxis=dict(range=[-0.5, 21.5], title="Length (m)"),
        yaxis=dict(range=[-0.1, 2.4], title="Height (m)", scaleanchor="x", scaleratio=1),
        height=420,
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "▶ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": int(35 / speed_multiplier), "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 0}
                    }]
                },
                {
                    "label": "⏸ Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]
                }
            ],
            "direction": "left",
            "x": 0.0, "y": 1.15
        }],
        sliders=[{
            "steps": [
                {"args": [[f.name], {"frame": {"duration": 0}, "mode": "immediate"}],
                 "label": f"{traj['t'][int(f.name)]*1000:.0f} ms",
                 "method": "animate"}
                for f in frames
            ],
            "x": 0.1, "len": 0.8, "y": -0.08
        }]
    )
    return fig


def create_topdown_animation(traj: Dict, speed_multiplier: float = 1.0):
    """Top-down pitch map with moving ball + trail."""
    t = traj["t"]
    x = traj["x"]
    y = traj["y"]
    n_frames = len(t)
    step = max(1, n_frames // 55)
    frames_idx = list(range(0, n_frames, step))
    if frames_idx[-1] != n_frames - 1:
        frames_idx.append(n_frames - 1)

    fig = go.Figure()

    # Pitch rectangle
    fig.add_shape(type="rect", x0=0, y0=-1.4, x1=20.12, y1=1.4,
                  line=dict(color="white", width=2), fillcolor="rgba(34,139,34,0.35)")
    # Crease lines
    fig.add_shape(type="line", x0=17.68, y0=-1.4, x1=17.68, y1=1.4, line=dict(color="white", width=1, dash="dot"))
    # Stumps
    fig.add_shape(type="line", x0=20.12, y0=-0.11, x1=20.12, y1=0.11, line=dict(color="yellow", width=5))

    fig.add_trace(go.Scatter(x=[x[0]], y=[y[0]], mode="markers",
                             marker=dict(size=13, color="red", line=dict(width=2, color="white")), name="Ball"))
    fig.add_trace(go.Scatter(x=[], y=[], mode="lines", line=dict(width=3, color="orange"), showlegend=False))

    frames = []
    for i in frames_idx:
        frames.append(go.Frame(
            data=[
                go.Scatter(x=[x[i]], y=[y[i]]),
                go.Scatter(x=x[:i+1], y=y[:i+1])
            ],
            name=str(i),
            traces=[0, 1]
        ))
    fig.frames = frames

    fig.update_layout(
        title="Top-Down Pitch Map – Live Ball Path",
        xaxis=dict(range=[-0.8, 21], title="Length (m)"),
        yaxis=dict(range=[-1.6, 1.6], title="Line (m)  (− = off side)", scaleanchor="x", scaleratio=1),
        height=420,
        plot_bgcolor="rgba(20,70,20,0.25)",
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {"label": "▶ Play", "method": "animate",
                 "args": [None, {"frame": {"duration": int(35 / speed_multiplier), "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]},
                {"label": "⏸ Pause", "method": "animate",
                 "args": [[None], {"frame": {"duration": 0}, "mode": "immediate"}]}
            ],
            "x": 0.0, "y": 1.15
        }]
    )
    return fig


# =============================================================================
# STREAMLIT UI
# =============================================================================

st.title("🏏 Multi-Modal Edge-AI Cricket Simulator")
st.caption("Technical Spec v4.0 · Animated Ball Flight · Edge-AI Fusion · Zero Cloud")

with st.sidebar:
    st.header("Session Controls")
    n_balls = st.slider("Deliveries in spell", 1, 18, 6)
    base_speed = st.slider("Base release speed (km/h)", 75, 150, 128)
    spin_type = st.selectbox("Bowling type", ["Pace", "Medium", "Off-Spin", "Leg-Spin"])
    st.markdown("---")
    st.subheader("Sensor Status")
    for s in ["Piezoelectric", "NIR Light Curtains", "Ultrasonic", "mmWave Radar", "6× OAK-D Spatial AI"]:
        st.success(f"{s}  ✓")
    st.markdown("---")
    if st.button("🔄 Run Full Fusion Pipeline", type="primary", use_container_width=True):
        st.session_state.run_sim = True

if "run_sim" not in st.session_state:
    st.session_state.run_sim = False
if "session_events" not in st.session_state:
    st.session_state.session_events = []

if st.session_state.run_sim:
    st.session_state.session_events = []
    bar = st.progress(0, text="Generating multi-modal streams + fusion…")
    for i in range(n_balls):
        speed = base_speed + random.uniform(-11, 11)
        length = random.uniform(3.8, 12.2)
        line = random.uniform(-0.65, 0.55)
        swing = random.uniform(-0.22, 0.22) if spin_type == "Pace" else random.uniform(-0.07, 0.07)
        seam = random.uniform(-0.16, 0.16)
        spin = {"Pace": 0, "Medium": 280, "Off-Spin": 1750, "Leg-Spin": 2150}[spin_type] + random.uniform(-180, 180)
        shot = random.choices(
            ["drive", "pull", "cut", "sweep", "leave", "defend"],
            weights=[0.22, 0.13, 0.15, 0.08, 0.27, 0.15]
        )[0]

        traj = generate_ball_trajectory(speed, length, line, swing, seam, spin_rpm=spin)
        sensors = simulate_sensors(traj, shot)
        fused = run_fusion_pipeline(sensors)
        st.session_state.session_events.append(fused)
        bar.progress((i + 1) / n_balls, text=f"Fused ball {i+1}/{n_balls}")
    st.session_state.run_sim = False
    bar.empty()
    st.success(f"Pipeline complete — {n_balls} deliveries fused (simulated edge latency < 18 ms)")

events = st.session_state.session_events

if events:
    # KPI row
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Avg Speed", f"{np.mean([e['traj']['release_speed_kmh'] for e in events]):.1f} km/h")
    valid = sum(e["valid_event"] for e in events)
    c2.metric("Valid Events", f"{valid}/{len(events)}")
    c3.metric("False Positives Rejected", f"{len(events)-valid}")
    c4.metric("Avg Fusion Conf.", f"{np.mean([e['analytics']['fusion_confidence'] for e in events])*100:.0f}%")
    c5.metric("Edge Latency", "< 18 ms")

    tab_anim, tab_bowl, tab_bat, tab_pipe = st.tabs([
        "🎬 Animated Ball Flight", "Bowling Analytics", "Batting Analytics", "Fusion Pipeline"
    ])

    with tab_anim:
        st.subheader("Realistic Ball Flight Replay")
        col_sel, col_spd = st.columns([2, 1])
        with col_sel:
            ball_idx = st.selectbox("Select delivery to replay",
                                    options=list(range(len(events))),
                                    format_func=lambda i: f"Ball {i+1} — {events[i]['traj']['release_speed_kmh']:.0f} km/h — {events[i]['analytics']['delivery']['length_category']}")
        with col_spd:
            speed_mult = st.select_slider("Playback speed", options=[0.5, 1.0, 1.5, 2.0, 3.0, 4.0], value=1.0)

        traj = events[ball_idx]["traj"]
        a1, a2 = st.columns(2)
        with a1:
            st.plotly_chart(create_side_on_animation(traj, speed_mult), use_container_width=True)
        with a2:
            st.plotly_chart(create_topdown_animation(traj, speed_mult), use_container_width=True)

        st.caption("Red ball = current position · Orange trail = path taken · Yellow = stumps · Use ▶ Play / slider to control")

    with tab_bowl:
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
                "Spin (rpm)": d["spin_rpm"]
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        speeds = [e["traj"]["release_speed_kmh"] for e in events]
        fig = px.line(x=list(range(1, len(speeds)+1)), y=speeds, markers=True,
                      labels={"x": "Ball", "y": "Release Speed (km/h)"},
                      title="Release Speed Consistency")
        st.plotly_chart(fig, use_container_width=True)

    with tab_bat:
        bat_rows = []
        for i, e in enumerate(events):
            b = e["analytics"]["batting"]
            bat_rows.append({
                "Ball": i+1,
                "Shot": b["shot_type"].title(),
                "Contact": b["contact_quality"],
                "Timing": b["timing"],
                "Footwork": b["footwork"],
                "Confidence": f"{e['analytics']['fusion_confidence']*100:.0f}%"
            })
        st.dataframe(pd.DataFrame(bat_rows), use_container_width=True, hide_index=True)

        shot_counts = pd.Series([e["analytics"]["batting"]["shot_type"] for e in events]).value_counts()
        fig = px.pie(values=shot_counts.values, names=shot_counts.index, title="Shot Type Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with tab_pipe:
        e = events[0]
        st.markdown(f"""
        **Stage A – Time Synchronization**  
        Master Piezo trigger used · ±25 ms window around impact

        **Stage B – Cross-Validation**  
        Valid event: `{e['valid_event']}` · False positives rejected: `{e['false_positive_rejected']}`

        **Stage C – High-Level ML**  
        Delivery: **{e['analytics']['delivery']['length_category']} / {e['analytics']['delivery']['line_category']}**  
        Shot: **{e['analytics']['batting']['shot_type']}** · Contact: **{e['analytics']['batting']['contact_quality']}**  
        Fusion confidence: **{e['analytics']['fusion_confidence']*100:.0f}%**
        """)
        st.info("All processing runs on the local Edge-AI box. Zero cloud dependency.")

else:
    st.info("Set the session parameters in the sidebar and click **Run Full Fusion Pipeline** to generate deliveries.")

st.markdown("---")
st.caption("Multi-Modal Edge-AI Sensor Fusion Simulator · Spec v4.0 · Animated Option A")

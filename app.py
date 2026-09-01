import streamlit as st
import fitparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import PowerNorm
from mplsoccer import Pitch

st.set_page_config(page_title="Football Heatmap", layout="centered")
st.title("⚽ Match Heatmap Dashboard")

# Venue Database
SAVED_VENUES = {
    'Albert_park_close_pitch': {
        'type': '9v9', 'length': 70, 'width': 45,
        'corners': [
            (53.5003, -2.2614), (53.5003, -2.2628),
            (53.4991, -2.2610), (53.4988, -2.2624)
        ]
    }
}

# --- UI Controls ---
uploaded_file = st.file_uploader("Upload Suunto .fit File", type=["fit"])

col1, col2 = st.columns(2)
with col1:
    is_caged = st.checkbox("Caged 5-a-Side Pitch", value=False)
with col2:
    selected_venue = st.selectbox("Or Select Saved Venue:", list(SAVED_VENUES.keys()), disabled=is_caged)

lap_option = st.selectbox("Laps to Include:", ["ALL", "1", "1, 2", "1, 3", "1, 4"])

if uploaded_file is not None:
    # 1. Parse FIT from memory buffer
    fitfile = fitparse.FitFile(uploaded_file.getvalue())

    # Get Lap Timestamps
    laps_info = []
    for lap in fitfile.get_messages('lap'):
        start_t, end_t = None, None
        for data in lap:
            if data.name == 'start_time':
                start_t = data.value
            elif data.name == 'timestamp':
                end_t = data.value
        if start_t and end_t:
            laps_info.append({'start': start_t, 'end': end_t})

    if lap_option == "ALL":
        active_lap_indices = list(range(len(laps_info)))
    else:
        chosen_laps = [int(x.strip()) for x in lap_option.split(",")]
        active_lap_indices = [i - 1 for i in chosen_laps if 0 <= i - 1 < len(laps_info)]

    # Extract Data
    lats, lons, speeds, distances = [], [], [], []
    for record in fitfile.get_messages('record'):
        rec_time, lat_val, lon_val, speed_val, dist_val = None, None, None, None, None
        for data in record:
            if data.name == 'timestamp':
                rec_time = data.value
            elif data.name == 'position_lat':
                lat_val = data.value
            elif data.name == 'position_long':
                lon_val = data.value
            elif data.name == 'speed':
                speed_val = data.value
            elif data.name == 'distance':
                dist_val = data.value

        if rec_time and lat_val and lon_val:
            if any(laps_info[idx]['start'] <= rec_time <= laps_info[idx]['end'] for idx in active_lap_indices):
                lats.append(lat_val * (180 / 2**31))
                lons.append(lon_val * (180 / 2**31))
                if speed_val is not None:
                    speeds.append(speed_val * 3.6)
                if dist_val is not None:
                    distances.append(dist_val)

    if len(lats) > 0:
        cutoff = int(len(lats) * 0.95)
        lats, lons = np.array(lats[:cutoff]), np.array(lons[:cutoff])
        speeds = np.array(speeds[:cutoff]) if speeds else np.array([0])

        total_dist_km = (distances[cutoff-1] - distances[0]) / 1000 if len(distances) > 0 else 0
        total_dist_miles = total_dist_km * 0.621371
        top_speed_kmh = np.max(speeds) if len(speeds) > 0 else 0
        top_speed_mph = top_speed_kmh * 0.621371
        avg_speed_kmh = np.mean(speeds[speeds > 1.0]) if len(speeds) > 0 else 0
        avg_speed_mph = avg_speed_kmh * 0.621371

        # Transform Coords
        if is_caged:
            p_length, p_width = 40, 20
            min_lon, max_lon = np.percentile(lons, [1, 99])
            min_lat, max_lat = np.percentile(lats, [1, 99])
            lons_clipped = np.clip(lons, min_lon, max_lon)
            lats_clipped = np.clip(lats, min_lat, max_lat)
            x_coords = (lons_clipped - min_lon) / (max_lon - min_lon) * p_length
            y_coords = (lats_clipped - min_lat) / (max_lat - min_lat) * p_width
        else:
            venue = SAVED_VENUES[selected_venue]
            p_length, p_width = venue['length'], venue['width']
            tl, tr, bl, br = venue['corners']
            src_pts = np.float32([[tl[1], tl[0]], [tr[1], tr[0]], [bl[1], bl[0]], [br[1], br[0]]])
            dst_pts = np.float32([[0, p_width], [p_length, p_width], [0, 0], [p_length, 0]])
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            gps_points = np.column_stack((lons, lats)).astype(np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(gps_points, M).reshape(-1, 2)
            x_coords, y_coords = transformed[:, 0], transformed[:, 1]
            
            PADDING = 2.0
            valid = (x_coords >= -PADDING) & (x_coords <= p_length + PADDING) & (y_coords >= -PADDING) & (y_coords <= p_width + PADDING)
            x_coords, y_coords = x_coords[valid], y_coords[valid]

        # Draw Pitch
        pitch = Pitch(pitch_type='custom', pitch_length=p_length, pitch_width=p_width, pitch_color='#12181b', line_color='#324148')
        fig, ax = pitch.draw(figsize=(10, 6.5))
        fig.patch.set_facecolor('#12181b')

        bin_stat = pitch.bin_statistic(x_coords, y_coords, statistic='count', bins=(60, 60))
        bin_stat['statistic'] = gaussian_filter(bin_stat['statistic'], sigma=1.2)
        pitch.heatmap(bin_stat, ax=ax, cmap='magma', norm=PowerNorm(gamma=0.35), edgecolors='none', alpha=0.85)

        title_text = "5-a-Side Match" if is_caged else f"Match at {selected_venue}"
        stats_text = f"Distance: {total_dist_km:.2f} km ({total_dist_miles:.2f} mi)  |  Top Speed: {top_speed_kmh:.1f} km/h ({top_speed_mph:.1f} mph)  |  Avg Speed: {avg_speed_kmh:.1f} km/h ({avg_speed_mph:.1f} mph)"

        ax.text(p_length/2, p_width + 3.0, title_text, color='white', fontsize=14, fontweight='bold', ha='center')
        ax.text(p_length/2, p_width + 1.2, stats_text, color='#00e676', fontsize=9, fontweight='bold', ha='center')

        st.pyplot(fig)

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

lap_input = st.text_input("Laps to Include (e.g. 1, 3 or ALL):", value="ALL")

if uploaded_file is not None:
    fitfile = fitparse.FitFile(uploaded_file.getvalue())

    # 1. Parse Lap Timestamps
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

    # Parse comma-separated lap input
    lap_input_clean = lap_input.strip().upper()
    if lap_input_clean == "ALL" or not lap_input_clean:
        active_lap_indices = list(range(len(laps_info)))
    else:
        try:
            chosen_laps = [int(x.strip()) for x in lap_input.split(",") if x.strip().isdigit()]
            active_lap_indices = [i - 1 for i in chosen_laps if 0 <= i - 1 < len(laps_info)]
        except ValueError:
            active_lap_indices = list(range(len(laps_info)))

    # 2. Extract Raw Record Data
    raw_records = []
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
            raw_records.append({
                'time': rec_time,
                'lat': lat_val * (180 / 2**31),
                'lon': lon_val * (180 / 2**31),
                'speed': speed_val * 3.6 if speed_val is not None else 0.0,
                'dist': dist_val if dist_val is not None else 0.0
            })

    if raw_records:
        start_time = raw_records[0]['time']
        total_duration_mins = int((raw_records[-1]['time'] - start_time).total_seconds() // 60)

        # Timeframe filter slider (in minutes)
        min_time, max_time = st.slider(
            "Select Match Timeframe (Minutes):",
            min_value=0,
            max_value=max(1, total_duration_mins),
            value=(0, max(1, total_duration_mins)),
            step=1
        )

        # 3. Filter by Laps and Selected Minute Range
        lats, lons, speeds, distances = [], [], [], []
        for r in raw_records:
            elapsed_mins = (r['time'] - start_time).total_seconds() / 60.0
            
            if not (min_time <= elapsed_mins <= max_time):
                continue

            if laps_info:
                in_active_lap = any(laps_info[idx]['start'] <= r['time'] <= laps_info[idx]['end'] for idx in active_lap_indices)
                if not in_active_lap:
                    continue

            lats.append(r['lat'])
            lons.append(r['lon'])
            speeds.append(r['speed'])
            distances.append(r['dist'])

        if len(lats) > 1:
            cutoff = int(len(lats) * 0.95)
            lats, lons = np.array(lats[:cutoff]), np.array(lons[:cutoff])
            speeds = np.array(speeds[:cutoff])

            total_dist_km = (distances[cutoff - 1] - distances[0]) / 1000 if len(distances) > 0 else 0
            total_dist_miles = total_dist_km * 0.621371
            top_speed_kmh = np.max(speeds) if len(speeds) > 0 else 0
            top_speed_mph = top_speed_kmh * 0.621371
            moving_speeds = speeds[speeds > 1.0]
            avg_speed_kmh = np.mean(moving_speeds) if len(moving_speeds) > 0 else 0
            avg_speed_mph = avg_speed_kmh * 0.621371

            # Transform Coords
            if is_caged:
                p_length, p_width = 40, 20
                min_lon, max_lon = np.percentile(lons, [1, 99])
                min_lat, max_lat = np.percentile(lats, [1, 99])
                lons_clipped = np.clip(lons, min_lon, max_lon)
                lats_clipped = np.clip(lats, min_lat, max_lat)
                x_coords = (lons_clipped - min_lon) / (max_lon - min_lon + 1e-7) * p_length
                y_coords = (lats_clipped - min_lat) / (max_lat - min_lat + 1e-7) * p_width
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

            # High-resolution 2D histogram + Gaussian smoothing
            n_bins_x = int(p_length * 4)
            n_bins_y = int(p_width * 4)
            heatmap_data, xedges, yedges = np.histogram2d(
                x_coords, y_coords,
                bins=[n_bins_x, n_bins_y],
                range=[[0, p_length], [0, p_width]]
            )
            heatmap_smoothed = gaussian_filter(heatmap_data, sigma=3.5)

            # Render smooth bicubic heatmap directly over the pitch
            ax.imshow(
                heatmap_smoothed.T,
                extent=[0, p_length, 0, p_width],
                origin='lower',
                cmap='magma',
                norm=PowerNorm(gamma=0.5),
                alpha=0.85,
                interpolation='bicubic',
                zorder=2
            )

            title_text = f"5-a-Side Match ({min_time}'-{max_time}')" if is_caged else f"Match at {selected_venue} ({min_time}'-{max_time}')"
            stats_text = f"Distance: {total_dist_km:.2f} km ({total_dist_miles:.2f} mi)  |  Top Speed: {top_speed_kmh:.1f} km/h ({top_speed_mph:.1f} mph)  |  Avg Speed: {avg_speed_kmh:.1f} km/h ({avg_speed_mph:.1f} mph)"

            ax.text(p_length / 2, p_width + 3.0, title_text, color='white', fontsize=14, fontweight='bold', ha='center')
            ax.text(p_length / 2, p_width + 1.2, stats_text, color='#00e676', fontsize=9, fontweight='bold', ha='center')

            st.pyplot(fig)
        else:
            st.warning("No GPS data found within the selected lap and minute window.")

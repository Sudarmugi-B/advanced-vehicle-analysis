import streamlit as st
import cv2
import numpy as np
import sqlite3
from pathlib import Path
import torch
from ultralytics import YOLO
import easyocr
import math
import time
from datetime import datetime
import pandas as pd
import plotly.express as px
from collections import defaultdict

# Add CSS for custom styling
def local_css():
    st.markdown("""
    <style>
    .stAlert {
        background-color: #ff4b4b;
        color: white;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize YOLO model and OCR reader
@st.cache_resource
def load_models():
    model = YOLO('vehicle_v10.pt')
    reader = easyocr.Reader(['en'])
    return model, reader

def init_db():
    conn = sqlite3.connect('vehicle_detection.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS vehicle_records
                 (timestamp TEXT, vehicle_type TEXT, direction TEXT, 
                  speed REAL, number_plate TEXT)''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS traffic_density
                 (timestamp TEXT, total_vehicles INTEGER, 
                  congestion_level TEXT)''')
    conn.commit()
    return conn

def calculate_speed(pos1, pos2, fps, frame_height):
    """
    Calculate vehicle speed with realistic constraints
    """
    try:
        # Calculate pixel distance
        distance = math.sqrt((pos2[0] - pos1[0])**2 + (pos2[1] - pos1[1])**2)
        
        # Scale factor: Assuming the frame height represents 20 meters in reality
        meters_per_pixel = 20.0 / frame_height
        
        # Calculate real distance in meters
        real_distance = distance * meters_per_pixel
        
        # Calculate time difference
        time_diff = 1/fps
        
        # Calculate speed in km/h
        speed = (real_distance/time_diff) * 3.6
        
        # Apply realistic constraints
        speed = min(speed, 60)  # Changed maximum speed to 60
        
        # Filter out unrealistic sudden changes
        if speed < 5:
            speed = 0
            
        return round(speed, 1)
        
    except Exception as e:
        print(f"Error calculating speed: {e}")
        return 0.0

def draw_detection_info(frame, x1, y1, x2, y2, speed, vehicle_type, plate_number, direction):
    """
    Enhanced visualization with better visibility
    """
    # Calculate box dimensions
    box_width = x2 - x1
    text_scale = box_width / 300
    text_scale = max(min(text_scale, 2.0), 0.7)
    
    # Box thickness based on frame size
    thickness = max(2, int(frame.shape[1] / 500))
    
    # Colors
    BOX_COLOR = (0, 255, 0)
    TEXT_COLOR = (255, 255, 255)
    BG_COLOR = (0, 0, 0)
    
    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, thickness)
    
    # Prepare text info
    info_text = [
        f"Type: {vehicle_type}",
        f"Speed: {speed} km/h",
        f"Direction: {direction}",
        f"Plate: {plate_number}"
    ]
    
    # Calculate text dimensions
    padding = 5
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_heights = []
    
    for text in info_text:
        (text_width, text_height), _ = cv2.getTextSize(text, font, text_scale, thickness)
        text_heights.append((text_width, text_height))
    
    # Draw background rectangle
    max_width = max(w for w, h in text_heights)
    total_height = sum(h for w, h in text_heights) + padding * (len(info_text) + 1)
    
    bg_x1 = max(0, x1)
    bg_y1 = max(0, y1 - total_height - padding)
    bg_x2 = min(frame.shape[1], x1 + max_width + padding * 2)
    bg_y2 = min(frame.shape[0], y1)
    
    cv2.rectangle(frame, 
                 (bg_x1, bg_y1), 
                 (bg_x2, bg_y2), 
                 BG_COLOR, 
                 -1)
    
    # Draw text
    y_offset = y1 - total_height
    for i, (text, (w, h)) in enumerate(zip(info_text, text_heights)):
        text_y = max(h + padding, y_offset + h + padding)
        cv2.putText(frame, text, 
                   (x1 + padding, text_y), 
                   font, text_scale, TEXT_COLOR, thickness)
        y_offset += h + padding

def analyze_traffic_density(total_vehicles, frame_area):
    """Calculate traffic density and congestion level"""
    density = total_vehicles / (frame_area / 1000000)  # vehicles per million pixels
    if density < 0.5:
        return "Low"
    elif density < 1.5:
        return "Medium"
    else:
        return "High"

def generate_analytics(conn):
    """Generate analytics from stored data"""
    df = pd.read_sql_query("""
        SELECT * FROM vehicle_records
        WHERE timestamp >= datetime('now', '-1 hour')
    """, conn)
    
    if not df.empty:
        # Vehicle type distribution
        fig_type = px.pie(df, names='vehicle_type', title='Vehicle Type Distribution')
        st.plotly_chart(fig_type)
        
        # Speed distribution with fixed y-axis range for vehicle count
        fig_speed = px.histogram(df, x='speed', 
                               title='Speed Distribution',
                               nbins=20)
        fig_speed.update_xaxes(title="Speed (km/h)")
        fig_speed.update_yaxes(title="Number of Vehicles", range=[0, 60])  # Set fixed range for vehicle count
        st.plotly_chart(fig_speed)

def process_frame(frame, model, reader, vehicle_tracker, prev_positions):
    results = model(frame)
    processed_frame = frame.copy()
    frame_height = frame.shape[0]
    frame_area = frame.shape[0] * frame.shape[1]
    
    detections = []
    vehicle_count_by_type = defaultdict(int)
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            if conf > 0.5 and cls in [2, 3, 5, 7]:  # car, motorcycle, bus, truck
                vehicle_id = len(detections)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                vehicle_type = model.names[cls]
                vehicle_count_by_type[vehicle_type] += 1
                
                # Calculate direction and speed
                if vehicle_id in prev_positions:
                    prev_center = prev_positions[vehicle_id]
                    speed = calculate_speed(prev_center, center, 30, frame_height)
                    direction = "Incoming" if center[1] > prev_center[1] else "Outgoing"
                else:
                    speed = 0
                    direction = "Unknown"
                
                # Number plate detection with retry
                plate_number = "Unknown"
                for attempt in range(2):
                    try:
                        expand = attempt * 10
                        y_start = max(0, y1 - expand)
                        y_end = min(frame.shape[0], y2 + expand)
                        x_start = max(0, x1 - expand)
                        x_end = min(frame.shape[1], x2 + expand)
                        plate_roi = frame[y_start:y_end, x_start:x_end]
                        
                        if plate_roi.size > 0:
                            plate_results = reader.readtext(plate_roi)
                            if plate_results:
                                plate_number = plate_results[0][1]
                                break
                    except Exception as e:
                        continue
                
                # Draw detection info
                draw_detection_info(processed_frame, x1, y1, x2, y2, speed, 
                                 vehicle_type, plate_number, direction)
                
                detections.append({
                    'id': vehicle_id,
                    'type': vehicle_type,
                    'direction': direction,
                    'speed': speed,
                    'plate': plate_number,
                    'center': center
                })
    
    # Add traffic density indicator
    congestion_level = analyze_traffic_density(len(detections), frame_area)
    cv2.putText(processed_frame, f"Traffic Density: {congestion_level}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return processed_frame, detections, vehicle_count_by_type, congestion_level

def main():
    st.title("Vehicle Analysis System")
    local_css()
    
    # Sidebar for settings
    st.sidebar.title("Settings")
    show_analytics = st.sidebar.checkbox("Show Real-time Analytics", True)
    
    # Load models
    model, reader = load_models()
    
    # Initialize database
    conn = init_db()
    
    # File uploader
    video_file = st.file_uploader("Upload a video file", type=['mp4', 'avi'])
    
    if video_file is not None:
        # Save uploaded file temporarily
        temp_path = Path("temp_video.mp4")
        with open(temp_path, 'wb') as f:
            f.write(video_file.read())
        
        # Video processing setup
        cap = cv2.VideoCapture(str(temp_path))
        
        # Create vertical layout
        st.subheader("Input Video")
        input_video = st.empty()
        
        st.subheader("Output Video")
        output_video = st.empty()
        
        # Metrics layout
        metrics_cols = st.columns(3)
        total_count = metrics_cols[0].empty()
        incoming_count = metrics_cols[1].empty()
        outgoing_count = metrics_cols[2].empty()
        
        # Analytics container (created once)
        if show_analytics:
            analytics_container = st.container()
        
        # Processing variables
        vehicle_tracker = {}
        prev_positions = {}
        last_analytics_update = time.time()
        analytics_update_interval = 5  # Update analytics every 5 seconds
        
        # Processing loop
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Display input frame
            input_video.image(frame, channels="BGR")
            
            # Process frame
            processed_frame, detections, vehicle_counts, congestion = process_frame(
                frame, model, reader, vehicle_tracker, prev_positions)
            
            # Update metrics
            total_vehicles = len(detections)
            incoming_vehicles = sum(1 for d in detections if d['direction'] == "Incoming")
            outgoing_vehicles = sum(1 for d in detections if d['direction'] == "Outgoing")
            
            # Display metrics
            total_count.metric("Total Vehicles", total_vehicles)
            incoming_count.metric("Incoming", incoming_vehicles)
            outgoing_count.metric("Outgoing", outgoing_vehicles)
            
            # Update database
            for detection in detections:
                conn.execute('''INSERT INTO vehicle_records VALUES (?, ?, ?, ?, ?)''',
                           (datetime.now().isoformat(), detection['type'],
                            detection['direction'], detection['speed'],
                            detection['plate']))
                
            conn.execute('''INSERT INTO traffic_density VALUES (?, ?, ?)''',
                        (datetime.now().isoformat(), total_vehicles, congestion))
            conn.commit()
            
            # Update trackers
            prev_positions = {d['id']: d['center'] for d in detections}
            
            # Display processed frame
            output_video.image(processed_frame, channels="BGR")
            
            # Update analytics periodically
            current_time = time.time()
            if show_analytics and (current_time - last_analytics_update) >= analytics_update_interval:
                with analytics_container:
                    st.subheader("Real-time Analytics")
                    generate_analytics(conn)
                last_analytics_update = current_time
            
            time.sleep(0.03)
        
        cap.release()
        temp_path.unlink()
    
    conn.close()

if __name__ == "__main__":
    main()
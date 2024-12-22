# Advanced Vehicle Analysis using YOLO v10

## Overview
This project is a comprehensive vehicle analysis system that leverages YOLO v10 for detection, counting, speed estimation, and number plate recognition. The application provides real-time analytics and traffic density visualization through an intuitive Streamlit-based interface.

## Features
- **Vehicle Detection:** Identifies cars, motorcycles, buses, and trucks.
- **Speed Estimation:** Calculates vehicle speed using realistic constraints.
- **Traffic Direction:** Classifies vehicles as `Incoming` or `Outgoing`.
- **Number Plate Recognition:** Detects and reads license plates using EasyOCR.
- **Traffic Density Analysis:** Provides congestion levels (`Low`, `Medium`, `High`).
- **Real-Time Analytics:** Displays vehicle type and speed distributions.

## Tech Stack
- **YOLO v10** for object detection.
- **EasyOCR** for optical character recognition.
- **Streamlit** for a user-friendly web interface.
- **SQLite** for database management.
- **OpenCV** for video processing and visualization.
- **Plotly** for interactive charts.

## Installation
### Prerequisites
Ensure you have Python 3.8+ and the following dependencies installed:

```bash
pip install -r requirements.txt
```

`requirements.txt` example:
```text
streamlit
opencv-python
numpy
sqlite3
torch
ultralytics
easyocr
pandas
plotly
```

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/Sudarmugi-B/advanced-vehicle-analysis
   cd your-repo-name
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place the YOLO model file (`vehicle_v10.pt`) in the project directory.

4. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage
1. Upload a video file (`mp4` or `avi`) using the web interface.
2. View real-time metrics for:
   - Total vehicle count
   - Incoming and outgoing vehicle counts
3. Analyze traffic density and congestion levels.
4. Access real-time analytics like vehicle type and speed distribution.

## File Structure
```
project-directory/
├── app.py                 # Main application script
├── requirements.txt       # Project dependencies
├── vehicle_v10.pt         # YOLO model file
├── vehicle_detection.db   # SQLite database
```

## Future Enhancements
- Incorporate additional vehicle types.
- Improve speed estimation accuracy using real-world GPS data.
- Enhance analytics with heatmaps for traffic flow.

## License
[MIT License](LICENSE)

## Acknowledgments
- [Ultralytics](https://github.com/ultralytics) for YOLO.
- [JaidedAI](https://github.com/JaidedAI/EasyOCR) for EasyOCR.

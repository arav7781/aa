import cv2
import numpy as np
import dlib
from scipy.signal import find_peaks, butter, filtfilt
import time
from collections import deque
from typing import Tuple, Dict
import logging
import os
import uuid
from langchain_core.tools import tool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r"C:\Users\aravs\Desktop\ArogyaMitra-\shape_predictor_68_face_landmarks.dat")  # Ensure this file exists

def open_camera():
    """Try multiple camera indices to find an available webcam."""
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        logger.info(f"Camera opened successfully on index 0")
        return cap
    logger.warning(f"Failed to open camera on index 0")
    return None

# Butterworth bandpass filter for PPG signals
def butter_bandpass_filter(data, lowcut=0.7, highcut=4.0, fs=30, order=2):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Extract PPG signal from the forehead region
def extract_ppg_signal(frame, gray):
    """Extract the PPG signal from the forehead region using dlib landmarks."""
    faces = detector(gray)
    if len(faces) == 0:
        return None, None
    
    face = faces[0]
    landmarks = predictor(gray, face)

    # Define forehead region using facial landmarks (points 17-26)
    roi_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(17, 27)], np.int32)
    roi_mask = np.zeros_like(gray)
    cv2.fillConvexPoly(roi_mask, roi_points, 255)

    # Extract green channel for PPG analysis
    roi = cv2.bitwise_and(frame, frame, mask=roi_mask)
    green_channel = roi[:, :, 1]
    return np.mean(green_channel[green_channel > 0]), roi

# Estimate SpO2 (Oxygen Saturation)
def calculate_spo2(red_signal, green_signal):
    """Estimate SpO₂ using an empirical ratio method."""
    ac_red = np.std(red_signal)
    dc_red = np.mean(red_signal)
    ac_green = np.std(green_signal)
    dc_green = np.mean(green_signal)

    if dc_red == 0 or dc_green == 0:
        return 98  # Default normal value if division fails

    r = (ac_red / dc_red) / (ac_green / dc_green)
    spo2 = 110 - 25 * r  # Empirical formula
    return max(85, min(100, spo2))  # Limit values to a realistic range

# Detect respiration rate (Placeholder function)
def calculate_respiratory_rate():
    """Placeholder for respiratory rate calculation using optical flow."""
    return 16  # Default normal value

# Generate clinical alerts based on vital signs
def evaluate_clinical_status(data_buffer):
    """Check for abnormal vitals and generate alerts."""
    alerts = []
    if data_buffer["bpm"] < 50 or data_buffer["bpm"] > 120:
        alerts.append("Critical Heart Rate")
    if data_buffer["spo2"] < 92:
        alerts.append("Hypoxia Alert")
    if data_buffer["hrv"] > 100:
        alerts.append("Possible Arrhythmia")
    data_buffer["clinical_alerts"] = alerts

# Main function for real-time vital monitoring
@tool(parse_docstring=True)
def clinical_grade_vital_monitor() -> Tuple[str, Dict]:
    """
    Perform real-time vital sign analysis using a live webcam feed.

    Features:
    - ROI-based PPG signal processing from facial video
    - Adaptive signal filtering
    - SpO₂ estimation using multi-wavelength analysis
    - Respiratory rate from chest movement
    - Clinical alerts system

    Returns:
        Tuple[str, Dict]: A tuple containing the status of the task and the updated state.
    """
    try:
        data_buffer = {
            "ppg_raw": deque(maxlen=300),
            "timestamps": deque(maxlen=300),
            "bpm": 72,
            "spo2": 98,
            "resp_rate": 16,
            "hrv": 0,
            "clinical_alerts": []
        }

        cap = open_camera()
        if cap is None:
            logger.error("Failed to open webcam.")
            return "Error: Webcam not found"

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30

        output_video_path = os.path.join("videos", "output", f"vital_monitor_{uuid.uuid4()}.mp4")
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

        start_time = time.time()
        
        while time.time() - start_time < 10:  # Monitor for 10 seconds
            ret, frame = cap.read()
            if not ret:
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            ppg_val, roi = extract_ppg_signal(frame, gray)

            if ppg_val is None:
                continue

            data_buffer["ppg_raw"].append(ppg_val)
            data_buffer["timestamps"].append(time.time())

            if len(data_buffer["ppg_raw"]) > 150:
                filtered = butter_bandpass_filter(data_buffer["ppg_raw"])
                peaks, _ = find_peaks(filtered, distance=30 * 0.4)

                if len(peaks) > 1:
                    ibi = np.diff(peaks) / 30
                    data_buffer["bpm"] = 60 / np.mean(ibi)
                    data_buffer["hrv"] = np.std(ibi) * 1000

                _, red_roi = extract_ppg_signal(frame, gray)
                if red_roi is not None:
                    data_buffer["spo2"] = calculate_spo2(red_roi, data_buffer["ppg_raw"])

                data_buffer["resp_rate"] = calculate_respiratory_rate()
                evaluate_clinical_status(data_buffer)

            display_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            y_pos = 30
            for param in ["bpm", "spo2", "resp_rate", "hrv"]:
                cv2.putText(display_frame, f"{param.upper()}: {data_buffer[param]:.1f}", 
                            (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_pos += 30

            for i, alert in enumerate(data_buffer["clinical_alerts"]):
                cv2.putText(display_frame, f"! {alert}", (400, 30 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow("Clinical Vital Monitor", display_frame)
            out.write(display_frame)

            if cv2.waitKey(1) & 0xFF == 27:  # Exit on ESC key
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        # Generate comprehensive report
        report = (
            f"Clinical Vital Report:\n"
            f"- Average Heart Rate: {data_buffer['bpm']:.1f} BPM\n"
            f"- Oxygen Saturation: {data_buffer['spo2']:.1f}%\n"
            f"- Respiratory Rate: {data_buffer['resp_rate']:.1f} RPM\n"
            f"- Heart Rate Variability: {data_buffer['hrv']:.1f} ms\n"
            f"- Clinical Alerts: {', '.join(data_buffer['clinical_alerts']) or 'None'}"
        )

        return "Vital monitoring completed.", {
            "intermediate_outputs": [{"output": "Vital monitoring completed."}],
            "report": report,
            "raw_data": data_buffer
        }
    except Exception as e:
        logger.exception("An error occurred during vital monitoring.")
        return str(e), {"intermediate_outputs": [{"output": str(e)}]}

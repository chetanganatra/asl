"""
Hand Gesture Data Collection Script
-----------------------------------

This script uses MediaPipe Hands + OpenCV to record labeled hand gesture data
into a CSV file for training gesture recognition models.

Best Practices Implemented:
1. Prompt user for gesture label at each run, saved into `label_landmark.csv`.
2. Records both hands consistently:
   - Left hand features first, then right hand.
   - If only one hand is detected, the other is filled with zeros.
3. Normalization of landmarks:
   - Translate all landmarks so wrist (landmark 0) is at (0,0,0).
   - Scale landmarks by wrist-to-middle-finger-MCP distance (hand size normalization).
4. Handedness confidence:
   - Only records samples where hand classification confidence ≥ 0.7.
   - Helps reduce noisy/incorrect data.
5. Frame sampling:
   - Records every 5th frame to avoid oversampling.
6. Debug info:
   - Prints detected hands, handedness, confidence, and when samples are saved.
7. Overlay info on video feed for real-time feedback.

Output CSV format:
label,
L0_x,L0_y,L0_z,...,L20_x,L20_y,L20_z,
R0_x,R0_y,R0_z,...,R20_x,R20_y,R20_z
"""

import cv2
import mediapipe as mp
import csv
import os
import math

# ============== Utility Functions ==============

def normalize_landmarks(landmarks):
    """
    Normalize landmarks:
    - Translate so wrist (landmark 0) is at origin (0,0,0).
    - Scale by distance wrist → middle finger MCP (landmark 9).
    Returns a flat list of normalized x,y,z coords.
    """
    wrist = landmarks[0]
    # Translation
    translated = [(lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z) for lm in landmarks]

    # Scale factor: wrist to middle finger MCP
    scale = math.sqrt((translated[9][0])**2 + (translated[9][1])**2 + (translated[9][2])**2)
    if scale < 1e-6:  # prevent division by zero
        scale = 1.0

    normalized = [(x/scale, y/scale, z/scale) for (x,y,z) in translated]
    return [coord for triple in normalized for coord in triple]  # flatten

def empty_hand():
    """Return zeros for one entire hand (21 landmarks × 3 = 63 values)."""
    return [0.0] * (21 * 3)

# ============== Main Script ==============

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Ask for gesture label
gesture_name = input("Enter gesture label (e.g., Hello, Yes, No): ").strip()
file_name = gesture_name+"_landmark.csv"

# Ensure CSV header
if not os.path.exists(file_name):
    with open(file_name, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["label"] + \
                 [f"L{i}_{axis}" for i in range(21) for axis in ("x","y","z")] + \
                 [f"R{i}_{axis}" for i in range(21) for axis in ("x","y","z")]
        writer.writerow(header)
    print(f"[INFO] Created new CSV file with header: {file_name}")

cap = cv2.VideoCapture(0)
frame_count = 0
save_count = 0

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        left_hand = None
        right_hand = None

        if results.multi_hand_landmarks and results.multi_handedness:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness.classification[0].label   # "Left" or "Right"
                score = handedness.classification[0].score

                # Debug print
                print(f"[DEBUG] Detected {label} hand with confidence {score:.2f}")

                if score < 0.7:
                    print("[DEBUG] Skipping low-confidence detection")
                    continue

                norm = normalize_landmarks(hand_landmarks.landmark)

                if label == "Left":
                    left_hand = norm
                else:
                    right_hand = norm

                # Draw hand landmarks
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        if frame_count % 5 == 0 and (results.multi_hand_landmarks and results.multi_handedness):
            # Fill missing hands with zeros, but only after we confirm at least one hand is present
            if left_hand is None:
                left_hand = empty_hand()
            if right_hand is None:
                right_hand = empty_hand()
            row = [gesture_name] + left_hand + right_hand
            with open(file_name, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
                save_count += 1
            print(f"[INFO] Saved sample {frame_count} for label '{gesture_name}' with total {save_count} samples")

        # Overlay text on frame
        cv2.putText(frame, f"Recording: {gesture_name}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Gesture Capture", frame)
        frame_count += 1

        # Quit with 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

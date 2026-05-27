import base64
import cv2
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from ultralytics import YOLO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'blind_nav_secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Load YOLO26 Nano model for optimized, ultra-low latency CPU inference
model = YOLO("yolo8n.pt")


def process_frame(img):
    height, width, _ = img.shape
    center_left = width // 3
    center_right = (width // 3) * 2

    # Run inference
    results = model(img, verbose=False)
    alerts = []

    for result in results:
        boxes = result.boxes
        for box in boxes:
            # Extract box properties
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            label = model.names[cls]

            # Target objects relevant to walking hazards (e.g., chair, person, box, car)
            if label in ['chair', 'person', 'table', 'car', 'bench', 'backpack', 'suitcase']:
                box_center_x = (x1 + x2) // 2

                # Simple heuristic: If the box takes up a massive portion of the lower screen, it's close!
                box_height_ratio = (y2 - y1) / height
                if box_height_ratio > 0.35:  # Adjust threshold for proximity sensitivity
                    # Determine obstacle zone location
                    if box_center_x < center_left:
                        zone = "on your left"
                    elif box_center_x > center_right:
                        zone = "on your right"
                    else:
                        zone = "directly ahead"

                    alerts.append(f"{label} {zone}")

    if alerts:
        # Prioritize the first critical threat found or combine them
        return f"Watch out! {alerts[0]}."
    return "Clear"


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Decode base64 image string from client
        header, encoded = data.split(",", 1)
        data_bytes = base64.b64decode(encoded)
        np_arr = np.frombuffer(data_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is not None:
            # Process image and get movement guidance
            instruction = process_frame(img)

            # Emit instruction back to the specific mobile client
            emit('navigation_instruction', {'message': instruction})
    except Exception as e:
        print(f"Error processing frame: {e}")


if __name__ == '__main__':
    # Host on 0.0.0.0 so your phone can connect via your laptop's local IP network
    socketio.run(app, host='0.0.0.0', port=5000, debug=True, ssl_context='adhoc')


# run terminal : python smartEyes_AI/app.py
# ip : https://192.168.52.104:5000
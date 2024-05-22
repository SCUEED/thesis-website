from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import cv2
import datetime
import os, sys
import numpy as np
from threading import Thread
import uvicorn
from starlette.middleware.cors import CORSMiddleware

# ==================================================================================

from roboflow import Roboflow
import supervision as sv
from PIL import Image
import base64
from inference import get_model
from tensorflow.keras.models import load_model
import shutil
import uuid
import string
import random
import time
import io
import pandas as pd
import base64

# ==================================================================================

model = get_model(model_id="apex-crackai/2", api_key="xPvuBOvC6PzAXmS2u6GF")

label_annotator = sv.LabelAnnotator()
mask_annotator = sv.MaskAnnotator()
bounding_box_annotator = sv.RoundBoxAnnotator()

crack_labels = {
  0: 'Hairline', 
  1: 'Non-Crack',
  2: 'Settlement',
  3: 'Structural'
}
title = "Real-time Detection"
confidence_score_binary = 0.45 # Less than 
confidence_score_detections = 0.60 # Greater than

# Load pre-trained EfficientNetB0 model
# crack_model = load_model('crack_detection_model_final.h5')

 
def resource_path(relative_path):
  """ Get absolute path to resource, works for dev and for PyInstaller """
  try:
    # PyInstaller creates a temp folder and stores path in _MEIPASS
    base_path = sys._MEIPASS
  except Exception:
    base_path = os.path.abspath(".")
 
  return os.path.join(base_path, relative_path)
 
 
crack_model = load_model(resource_path('./crack_classifier_model_final.h5'))
crack_binary_model = load_model(resource_path('./crack_detection_vgg16_model.h5'))
output_folder = resource_path("./output_folder")


# ==================================================================================


# Global variables for capture and switch
capture = 0
switch = 1

# Make shots directory to save pictures
os.makedirs('./shots', exist_ok=True)

# Instantiate FastAPI app
app = FastAPI()

# Enable CORS for the frontend server
origins = [
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Initialize the camera
camera = cv2.VideoCapture(0)

# ==================================================================================


# Function to preprocess the frame
def preprocess_frame(frame):
  frame = cv2.resize(frame, (224, 224))
  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
  frame = frame.astype(np.float32) / 255.0  # Convert to float32 and normalize to [0, 1] range
  frame = np.expand_dims(frame, axis=0)  # Add batch dimension
  return frame

def labelers(item, image_array):
    # Calculate the top-left corner coordinates (x, y) based on center (item.x, item.y),
    # width (item.width), and height (item.height) of the bounding box
    x1 = int(max(0, item.x - item.width / 2))  # Calculate left edge
    y1 = int(max(0, item.y - item.height / 2))  # Calculate top edge

    # Calculate the bottom-right corner coordinates (x2, y2)
    x2 = int(min(image_array.shape[1], x1 + item.width))
    y2 = int(min(image_array.shape[0], y1 + item.height))
    
    cropped_image = image_array[y1:y2, x1:x2]
  
    frame_preprocessed = preprocess_frame(cropped_image)
    frame_preprocessed = frame_preprocessed.astype(np.float32)
    
    # Classification
    predictions = crack_model.predict(frame_preprocessed)[0]
    predicted_class_index = np.argmax(predictions)
    
    # Binary
    prediction_binary = crack_binary_model.predict(frame_preprocessed)[0]
    is_crack = prediction_binary[0] < confidence_score_binary

    return crack_labels[predicted_class_index], is_crack

def save_frame_as_image(frame):
    unique_id = uuid.uuid4().hex[:10]
    timestamp = int(time.time())
    file_name = f"{unique_id}_{timestamp}.png"
    file_path = os.path.join(output_folder, file_name)
    image = Image.fromarray(frame, mode='RGB')
    image.save(file_path)
    return file_path

def image_file_to_base64(file_path):
    with open(file_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return encoded_string



# ==================================================================================

# Font settings
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_thickness = 2

global isReporting
global reports

isReporting = False
reports = []

# ==================================================================================


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index_2.html", {"request": request})

# ==================================================================================

@app.get('/video_feed')
async def video_feed():
    def generate():
        global annotated_image
        global ret
        global isReporting
        global reports

        while True: 
            ret, frame = camera.read()

            copy_frame = frame.copy()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            
            counting = {
                'Non-Crack': 0,
                'Hairline': 0,
                'Settlement': 0,
                'Structural': 0
            }    

            results = model.infer(frame)
            filtered_detections, labels = [], []
            for preds in results[0].predictions:
                pred, is_crack = labelers(preds, frame)

                if is_crack and pred != 'Non-Crack' and preds.confidence >= confidence_score_detections:
                    filtered_detections.append(preds)
                    labels.append(pred)
                    counting[pred] += 1

            results[0].predictions = filtered_detections
            detections = sv.Detections.from_inference(results[0].dict(by_alias=True, exclude_none=True))

            annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
            annotated_image = mask_annotator.annotate(scene=frame, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            text_small = f"Hairline: {counting['Hairline']}"
            text_medium = f"Settlement: {counting['Settlement']}"
            text_large = f"Structural: {counting['Structural']}"


            # Position for the text
            start_small = (50, 50)
            start_medium = (50, 100)
            start_large = (50, 150)

             # Add text to the image
            cv2.putText(annotated_image, text_small, start_small, font, font_scale, (206, 5, 103), font_thickness)
            cv2.putText(annotated_image, text_medium, start_medium, font, font_scale, (206, 5, 103), font_thickness)
            cv2.putText(annotated_image, text_large, start_large, font, font_scale, (206, 5, 103), font_thickness)

            if isReporting: reports.append(
                {
                    'Hairline': counting['Hairline'],
                    'Settlement':counting['Settlement'],
                    'Structural': counting['Structural'],
                    'annotated_image': save_frame_as_image(annotated_image),
                    'image': save_frame_as_image(copy_frame),
                    'datetime': datetime.datetime.now(),
                }
            )

            # Encode the frame as JPEG
            ret, buffer = cv2.imencode('.jpg', annotated_image)
            
            if not ret:
                continue
            frame = buffer.tobytes()



            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return StreamingResponse(generate(), media_type='multipart/x-mixed-replace; boundary=frame')

# ==================================================================================

# @app.post("/start_report", response_class=JSONResponse)
# async def start_report(request: Request):
#     data = await request.json()  # Get JSON data
#     if data.get("click") == "Start_Report":
#         isReporting = True
#         report = []
#         return {"message": "Image capture initiated"}
#     else:
#         return {"message": "Invalid request"}
    

# @app.post("/stop_report", response_class=JSONResponse)
# async def stop_report(request: Request):
#     data = await request.json()  # Get JSON data
#     if data.get("click") == "Stop_Report":
#         isReporting = False
#         result, report = report, []
#         return {"data": result}
#     else:
#         return {"message": "Invalid request"}

report_output_folder = resource_path("./report_output")

def csv_files_to_array_of_objects(directory):
    files = os.listdir(directory)
    csv_files = [file for file in files if file.endswith('.csv')][::-1]
    all_data = []
    for file in csv_files:
        file_path = os.path.join(directory, file)
        df = pd.read_csv(file_path)
        data = df.to_dict(orient='records')
        for i in range(len(data)):
          print(data[i]['image'])
          data[i]['annotated_image'] = image_file_to_base64(data[i]['annotated_image'])
          data[i]['image'] = image_file_to_base64(data[i]['image'])
        all_data.append({'report': data, 'name': file, 'date': datetime.datetime.now()})
    return all_data

@app.get("/get_reports", response_class=JSONResponse)
async def get_r(request: Request):
  reports = csv_files_to_array_of_objects(report_output_folder)
  return {"data": reports}

@app.post("/start_report", response_class=JSONResponse)
async def start_report(request: Request):
  global isReporting
  global reports
  isReporting = True
  reports = []
  return {"message": "Image capture initiated"}
    

@app.post("/stop_report", response_class=JSONResponse)
async def stop_report(request: Request):
  global isReporting
  global reports
  isReporting = False
  result = reports
  reports = []
  entries = os.listdir(report_output_folder)
  file_name = f'{len(entries)}_reports.csv'
  df = pd.DataFrame(result)
  df.to_csv(f'{os.path.join(report_output_folder, file_name)}')
  return {"data": result}

    



# @app.post("/requests", response_class=JSONResponse)
# async def tasks(background_tasks: BackgroundTasks, request: Request):
#     data = await request.json()  # Get JSON data
#     if data.get("click") == "Capture":
#         background_tasks.add_task(capture_image)  # Run in the background
#         return {"message": "Image capture initiated"}
#     else:
#         return {"message": "Invalid request"}

# def capture_image(ret, annotated_image):  # Function to capture image
#     global capture
#     if ret:
#          if capture:
#                 capture = 0
#                 now = datetime.datetime.now()
#                 p = os.path.sep.join(['shots', f"REPORT_ANNOTATED_{now.strftime('%Y%m%d_%H%M%S')}.jpg"])
#                 cv2.imwrite(p, annotated_image)

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
    camera.release()
    cv2.destroyAllWindows()

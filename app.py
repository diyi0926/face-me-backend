#!/usr/bin/env python
from flask import Flask, render_template, Response
import numpy as np
import cv2
import base64
from flask_socketio import SocketIO, emit, send
from time import sleep
from threading import Thread
import uuid
from get_frame import detect_faces, generate_frontal_face
from face_match import FaceMatcher

THREAD = Thread()
DET_THREAD = Thread()

APP = Flask(__name__)
SIO = SocketIO(APP)

CAM = cv2.VideoCapture(0)

# @APP.route('/')
# def index():
#     return render_template('index.html')

# @APP.route('/video_feed')
# def video_feed():
#     return Response(gen(CAM),
#                     mimetype='multipart/x-mixed-replace; boundary=frame')

# def gen(camera):
#     while camera.isOpened():
#         success, frame = camera.read()
#         ret, jpeg = cv2.imencode('.jpg', frame)
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
#     CAM.release()

""" Live Stream Functions and Thread"""


class VideoStreamThread(Thread):
    """Stream data on thread"""
    def __init__(self):
        self.delay = .06
        super(VideoStreamThread, self).__init__()

    def get_data(self):
        """
        Get data and emit to socket
        """
        global CAM
        while CAM.isOpened():
            _, frame = CAM.read()
            _, jpeg = cv2.imencode('.jpg', frame)
            encoded_img = "data:image/jpg;base64," + str(base64.b64encode(jpeg.tobytes()).decode())
            SIO.emit('video_frame',
                     {'frame': encoded_img},
                     namespace='/live-stream')
            sleep(self.delay)
            
    def run(self):
        """Default run method"""
        self.get_data()


@SIO.on('connect', namespace='/live-stream')
def connect_socket():
    """Handle socket connection"""
    global THREAD

    # Start thread
    if not THREAD.isAlive():
        THREAD = VideoStreamThread()
        THREAD.start()


class DetectionStreamThread(Thread):
    """Stream data on thread"""
    def __init__(self):
        self.delay = 0
        super(DetectionStreamThread, self).__init__()

    def get_data(self):
        """
        Get data and emit to socket
        """
        global CAM
        count = 0
        while CAM.isOpened():
            count += 1
            print('COUNT' + str(count))
            _, frame = CAM.read()

            # cropped face
            cropped_face, bbox_coordinate, anchor_coordinate = detect_faces(frame)
            if cropped_face is None:
                print("NONE FACE DETECTED")
                sleep(1)
                continue

            # get fake face
            fake_face, profile_feature_vector = generate_frontal_face(cropped_face)

            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
            fake_face = cv2.cvtColor(fake_face, cv2.COLOR_BGR2RGB)

            # face matching
            face_matcher = FaceMatcher()
            matched_face, matched_name, matched_front_fake_face, matched_diff = \
                face_matcher.match(cropped_face, fake_face, profile_feature_vector)

            matched_face = cv2.cvtColor(matched_face, cv2.COLOR_BGR2RGB)
            matched_front_fake_face = cv2.cvtColor(matched_front_fake_face, cv2.COLOR_BGR2RGB)

            _, cropped_face_jpeg = cv2.imencode('.jpg', cropped_face)
            _, fake_face_jpeg = cv2.imencode('.jpg', fake_face)
            _, matched_face_jpeg = cv2.imencode('.jpg', matched_face)
            _, matched_front_fake_face_jpeg = cv2.imencode('.jpg', matched_front_fake_face)

            encoded_cropped_face = "data:image/jpg;base64," + str(
                base64.b64encode(cropped_face_jpeg.tobytes()).decode())
            encoded_fake_face = "data:image/jpg;base64," + str(
                base64.b64encode(fake_face_jpeg.tobytes()).decode())

            encoded_matched_face = "data:image/jpg;base64," + str(
                base64.b64encode(matched_face_jpeg.tobytes()).decode())
            encoded_matched_front_fake_face = "data:image/jpg;base64," + str(
                base64.b64encode(matched_front_fake_face_jpeg.tobytes()).decode())

            # get detection model return here and send to face frontalization model
            SIO.emit('detection', {'cropped_face': encoded_cropped_face,
                                   'fake_face': encoded_fake_face,
                                   'matched_face': encoded_matched_face,
                                   'matched_name': matched_name,
                                   'matched_front_fake_face': encoded_matched_front_fake_face,
                                   'id': uuid.uuid4().hex},
                     namespace='/detections')
            sleep(self.delay)

    def run(self):
        """Default run method"""
        self.get_data()


@SIO.on('connect', namespace='/detections')
def start_detection_stream():
    """Handle socket connection"""
    global DET_THREAD

    # Start thread
    if not DET_THREAD.isAlive():
        DET_THREAD = DetectionStreamThread()
        DET_THREAD.start()

if __name__ == '__main__':
    SIO.run(APP)
    # app.run(host='0.0.0.0', debug=True)

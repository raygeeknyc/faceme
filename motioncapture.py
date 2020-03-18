#!/usr/bin/python3
# above is not used on macOS
_Pi = True
_Pi = False

import logging
logging.getLogger().setLevel(logging.INFO)

# Import the packages we need for drawing and displaying images
from PIL import Image, ImageDraw

from random import randint
import io
import sys
import os
import threading
import time
import signal
import queue
import numpy
import cv2


# This is how long to capture frames in the demo main application
DEMO_CAPTURE_SECS = 3.0

# This is the desired resolution of the camera
RESOLUTION = (320, 240)
# This is the desired maximum frame capture rate of the camera
CAPTURE_RATE_FPS = 10.0
# This value was determined from over an observed covered camera's noise
TRAINING_SAMPLES = 5
# This is how much the green channel has to change to consider a pixel changed
PIXEL_SHIFT_SENSITIVITY = 50
# This is the portion of pixels to compare when detecting motion
MOTION_DETECT_SAMPLE = 1.0/10  # so... 10%? (Kudos to Sarah Cooper)

# This is how long to sleep in various threads between shutdown checks
POLL_SECS = 0.1

FRAME_LATENCY_WINDOW_SIZE_SECS = 1.0

def calculate_image_difference(image_1, image_2, tolerance=None, sample_percentage=MOTION_DETECT_SAMPLE, record_delta=False):
    "Detect changes in the green channel."
    delta_pixels = None
    if record_delta:
        delta_pixels = []
    s=time.time()
    changed_pixels = 0
    pixel_step = int((RESOLUTION[0] * RESOLUTION[1])/(MOTION_DETECT_SAMPLE * RESOLUTION[0] * RESOLUTION[1]))
    current_pixels = image_2.reshape((RESOLUTION[0] * RESOLUTION[1]), 3)
    prev_pixels = image_1.reshape((RESOLUTION[0] * RESOLUTION[1]), 3)
    for pixel_index in range(0, RESOLUTION[0]*RESOLUTION[1], pixel_step):
        if abs(int(current_pixels[pixel_index][1]) - int(prev_pixels[pixel_index][1])) > PIXEL_SHIFT_SENSITIVITY:
            changed_pixels += 1
            if record_delta:
                delta_pixels.append(pixel_index)
            if tolerance and changed_pixels > tolerance:
              logging.debug("Image diff short circuited at: {}".format(time.time() - s))
              return (changed_pixels, delta_pixels)
    logging.debug("Image diff took: {}".format(time.time() - s))
    return (changed_pixels, delta_pixels)


class ImageCapture(object):
    def __init__(self, is_camera_pi, key_frame_queue):
        self._stop = False
        self._set_camera_type(is_camera_pi)
        self._key_frame_queue = key_frame_queue
        self._last_frame_at = 0.0
        self._frame_delay_secs = 1.0/CAPTURE_RATE_FPS
        self._current_frame_seq = 0

    def stop(self):
        logging.info("stop()")
        self._stop = True

    def _set_camera_type(self, use_pi_camera):
        if use_pi_camera:
            self._frame_provider = PiCamera()
        else:
            self._frame_provider = WebCamera()

    def get_next_frame(self):
        delay = (self._last_frame_at + self._frame_delay_secs) - time.time()
        if delay > 0:
            logging.debug("frame delay: {}".format(delay))
            time.sleep(delay)
        self._current_frame = self._frame_provider.get_frame()
        self._last_frame_at = time.time()
        self._current_frame_seq += 1

    def is_image_difference_over_threshold(self, changed_pixels_threshold):
        changed_pixels, _ = calculate_image_difference(self._prev_frame, self._current_frame, tolerance=changed_pixels_threshold)
        return changed_pixels > changed_pixels_threshold

    def _train_motion(self):
        logging.debug("Training motion")
        trained = False
        try:
            self._motion_threshold = 9999
            self.get_next_frame()
            for i in range(TRAINING_SAMPLES):
                self._prev_frame = self._frame_provider.get_frame()
                self.get_next_frame()
                motion, _ = calculate_image_difference(self._prev_frame, self._current_frame)
                self._motion_threshold = min(motion, self._motion_threshold)
            trained = True
        except Exception as e:
            logging.exception("Error training motion")
        logging.debug("Trained {}".format(trained))
        return trained

    def configure_capture(self):
        self._frame_provider._init_camera()
        if not self._attempt_motion_training():
            logging.error("Unable to train motion, exiting.")
            return False
        return True

    def _attempt_motion_training(self):
        logging.debug("Training motion detection")
        for retry in range(3):
            if self._train_motion():
                logging.info("Trained motion detection {}".format(self._motion_threshold))
                return True
        return False

    def capture_frames(self):
        self._current_frame_seq = 0
        logging.debug("capturing frames")
        self.get_next_frame()  # To give the initial motion detection a baseline
        while not self._stop:
           self._prev_frame = self._current_frame
           self.get_next_frame()
           if self.is_image_difference_over_threshold(self._motion_threshold):
               logging.debug("Motion detected")
               self._key_frame_queue.put((self._current_frame_seq, self._current_frame))

    def _cleanup(self):
        logging.debug("closing key frame queue")
        self._key_frame_queue.close()

class WebCamera(object):
  def _init_camera(self):
    import cv2

    logging.info("Using WebCam for video capture")
    self._camera = cv2.VideoCapture(0)
    if not self._camera.isOpened():
      logging.error("Video camera not opened")
      sys.exit(-1)

    self._camera.set(3, RESOLUTION[0])
    self._camera.set(4, RESOLUTION[1])

  def get_frame(self):
      _, frame = self._camera.read()
      return frame

  def _close_video(self):
      self._camera.release()

class PiCamera(object):
  def _init_camera(self):
    from picamera import PiCamera
    from picamera.array import PiRGBArray

    logging.info("Using PiCamera for video capture")
    self._camera = PiCamera()
    self._camera.resolution = RESOLUTION
    self._camera.vflip = False
    self._camera.framerate = 32
    self._image_buffer = io.BytesIO()

  def get_frame(self):
      self._camera.capture(self._image_buffer, format="jpeg", use_video_port=True)
      self._image_buffer.seek(0)
      data = numpy.fromstring(self._image_buffer.getvalue(), dtype=numpy.uint8)
      image = cv2.imdecode(data, 1)
      image = image[:, :, ::-1]
      return image

  def _close_video(self):
      self._camera.close()

def generate_delta_image(frames):
    logging.info("frames %d,%d", frames[0][0], frames[1][0])
    _, delta_pixels = calculate_image_difference(frames[0][1], frames[1][1], tolerance=None, sample_percentage=100, record_delta=True)
    logging.info("%d pixels changed", len(delta_pixels))

def collect_key_frame_pairs(key_frames, new_frame):
    key_frames.append(new_frame)
    if len(key_frames) == 2:
        generate_delta_image(key_frames)
        key_frames = []
    return key_frames

def main():
    logging.info("running %s", sys.argv[0])
    key_frame_queue = queue.Queue()
    frame_capturer = ImageCapture(_Pi, key_frame_queue)
    frame_capturer.configure_capture()
    frame_source = threading.Thread(target=frame_capturer.capture_frames)
    frame_source.start()
    start_capture = time.time()
    end_capture = start_capture + DEMO_CAPTURE_SECS
    key_frames = []
    num_frames = 0
    while time.time() < end_capture:
        if not key_frame_queue.empty():
            num_frames += 1
            key_frames = collect_key_frame_pairs(key_frames, key_frame_queue.get())
    frame_capturer.stop()
    frame_source.join()
    logging.info("%d frames at producer stop", num_frames)
    while not key_frame_queue.empty():
        num_frames += 1
        key_frames = collect_key_frame_pairs(key_frames, key_frame_queue.get())
        
    logging.info("captured %d key frames in %f seconds", num_frames, DEMO_CAPTURE_SECS)
    logging.info("exiting %s", sys.argv[0])
    sys.exit(0)

if __name__ == "__main__":
    main()

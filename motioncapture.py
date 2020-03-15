_Pi = False
_Pi = True

import logging

# Import the packages we need for drawing and displaying images
from PIL import Image, ImageDraw

import multiprocessing
from multiprocessingloghandler import ChildMultiProcessingLogHandler, ParentMultiProcessingLogHandler
from random import randint
import io
import sys
import os
import time
import signal
import Queue
import numpy
import cv2

# This is the desired resolution of the camera
RESOLUTION = (320, 240)
# This is the desired maximum frame capture rate of the camera
CAPTURE_RATE_FPS = 5.0
# This value was determined from over an observed covered camera's noise
TRAINING_SAMPLES = 5
# This is how much the green channel has to change to consider a pixel changed
PIXEL_SHIFT_SENSITIVITY = 30
# This is the portion of pixels to compare when detecting motion
MOTION_DETECT_SAMPLE = 1.0/10  # so... 10%? (Kudos to Sarah Cooper)

# This is how long to sleep in various threads between shutdown checks
POLL_SECS = 0.1

FRAME_LATENCY_WINDOW_SIZE_SECS = 1.0

EMPTY_LABELS = []

class ImageProducer(multiprocessing.Process):
    def __init__(self, event, key_frame_queue, log_queue, logging_level):
        multiprocessing.Process.__init__(self)
        self._exit = event
        self._log_queue = log_queue
        self._logging_level = logging_level
        self._key_frame_queue = key_frame_queue
        self._last_frame_at = 0.0
        self._frame_delay_secs = 1.0/CAPTURE_RATE_FPS
        self._current_frame_seq = 0
        self._frame_window_start = 0
        self._frame_latency_window_start = 0

    def _init_logging(self):
        handler = ChildMultiProcessingLogHandler(self._log_queue)
        logging.getLogger(str(os.getpid())).addHandler(handler)
        logging.getLogger(str(os.getpid())).setLevel(self._logging_level)

    def get_next_frame(self):
        delay = (self._last_frame_at + self._frame_delay_secs) - time.time()
        if delay > 0:
            logging.debug("frame delay: {}".format(delay))
            time.sleep(delay)
        self._current_frame = self._get_frame()
        self._last_frame_at = time.time()
        self._current_frame_seq += 1
        if time.time() > (self._frame_latency_window_start + FRAME_LATENCY_WINDOW_SIZE_SECS):
          window_fps = (self._current_frame_seq - self._frame_window_start)/(time.time() - self._frame_latency_window_start)
          logging.debug("Frame {}, window {} in {} secs, fp/s: {}, delay: {}".format(self._current_frame_seq, (self._current_frame_seq - self._frame_window_start), (time.time() - self._frame_latency_window_start), window_fps, self._frame_delay_secs))
          self._frame_window_start = self._current_frame_seq
          self._frame_latency_window_start = time.time()

    def calculate_image_difference(self, tolerance=None, sample_percentage=MOTION_DETECT_SAMPLE):
        "Detect changes in the green channel."
        s=time.time()
        changed_pixels = 0
        pixel_step = int((RESOLUTION[0] * RESOLUTION[1])/(MOTION_DETECT_SAMPLE * RESOLUTION[0] * RESOLUTION[1]))
        current_pixels = self._current_frame.reshape((RESOLUTION[0] * RESOLUTION[1]), 3)
        prev_pixels = self._prev_frame.reshape((RESOLUTION[0] * RESOLUTION[1]), 3)
        for pixel_index in xrange(0, RESOLUTION[0]*RESOLUTION[1], pixel_step):
            if abs(int(current_pixels[pixel_index][1]) - int(prev_pixels[pixel_index][1])) > PIXEL_SHIFT_SENSITIVITY:
                changed_pixels += 1
                if tolerance and changed_pixels > tolerance:
                  logging.debug("Image diff short circuited at: {}".format(time.time() - s))
                  return changed_pixels
        logging.debug("Image diff took: {}".format(time.time() - s))
        return changed_pixels

    def is_image_difference_over_threshold(self, changed_pixels_threshold):
        changed_pixels = self.calculate_image_difference(changed_pixels_threshold)
        return changed_pixels > changed_pixels_threshold

    def _train_motion(self):
        logging.debug("Training motion")
        trained = False
        try:
            self._motion_threshold = 9999
            self.get_next_frame()
            for i in range(TRAINING_SAMPLES):
                self._prev_frame = self._get_frame()
                self.get_next_frame()
                motion = self.calculate_image_difference()
                self._motion_threshold = min(motion, self._motion_threshold)
            trained = True
        except Exception, e:
            logging.exception("Error training motion")
            trained = False
            sys.exit(1)
        logging.debug("Trained {}".format(trained))
        return trained

    def run(self):
        self._init_logging()
        logging.debug("Image producer running")
        self._init_camera()
        self._attempt_motion_training()
        try:
            self._capture_frames()
        except Exception, e:
            logging.exception("Error in vision main thread")
        finally:
            self._cleanup()
            logging.debug("Exiting image capture")

    def _init_camera(self):
        logging.error("overide _init_camera()")

    def _attempt_motion_training(self):
        logging.debug("Training motion detection")
        for retry in xrange(3):
            if self._train_motion():
                break
        logging.info("Trained motion detection {}".format(self._motion_threshold))

    def _capture_frames(self):
        logging.debug("capturing frames")
        try:
            self.get_next_frame()
            while not self._exit.is_set():
                self._prev_frame = self._current_frame
                self.get_next_frame()
                if self.is_image_difference_over_threshold(self._motion_threshold):
                    logging.debug("Motion detected")
                    self._key_frame_queue.put((self._current_frame_seq, self._current_frame))
        except Exception, e:
            logging.exception("Error in capture_frames")
        logging.debug("Exiting vision capture thread")

    def _cleanup(self):
        logging.debug("closing key frame queue")
        self._key_frame_queue.close()

class WebcamImageProducer(ImageProducer):
  def _init_camera(self):
    self._camera = cv2.VideoCapture(0)

    if not self._camera.isOpened():
      logging.error("Video camera not opened")
      sys.exit(255)

    self._camera.set(3, RESOLUTION[0])
    self._camera.set(4, RESOLUTION[1])


  def _get_frame(self):
      _, frame = self._camera.read()
      return frame

  def _close_video(self):
      self._camera.release()

class PiImageProducer(ImageProducer):
  def _init_camera(self):
    self._camera = PiCamera()
    self._camera.resolution = RESOLUTION
    self._camera.vflip = False
    self._camera.framerate = 32
    self._image_buffer = io.BytesIO()

  def _get_frame(self):
      self._camera.capture(self._image_buffer, format="jpeg", use_video_port=True)
      self._image_buffer.seek(0)
      data = numpy.fromstring(self._image_buffer.getvalue(), dtype=numpy.uint8)
      image = cv2.imdecode(data, 1)
      image = image[:, :, ::-1]
      return image

  def _close_video(self):
      self._camera.close()

if _Pi:
  logging.debug("Using PiCamera for video capture")
  from picamera import PiCamera
  from picamera.array import PiRGBArray
  frame_provider = PiImageProducer
else:
  import cv2
  frame_provider = WebcamImageProducer

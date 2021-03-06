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
CAPTURE_RATE_FPS = 5.0
# This value was determined from over an observed covered camera's noise
TRAINING_SAMPLES = 5
# This is how much the green channel has to change to consider a pixel changed
PIXEL_SHIFT_SENSITIVITY = 50
# This is the portion of pixels to compare when detecting motion
MOTION_DETECT_SAMPLE_PERCENTAGE = 5  # so... 1 twentieth? (Kudos to Sarah Cooper)

# This is the amount to dim the at rest portions of a key frame in the demo
PIXEL_DIMMING_PERCENTAGE = 60


def calculate_image_delta(image_1, image_2, tolerance=None, sample_percentage=MOTION_DETECT_SAMPLE_PERCENTAGE, record_delta=False):
    """
    Calculate the number of pixels that differ in the green channel.
    Args:
        image_1: The first of a pair of successive Frames
        image_2: The second of a pair of successive Frames
        tolerance: A trigger threshold for the number of pixels, once this is reached we exit without looking further
        sample_percentage: The percentage of the image pixels to compare, we sample this percentage evenly distributed
        record_delta: If True, record the offset of each changed pixel and return the list
    Returns:
        changed_pixel_count: The number of changed pixels detected, If tolerance, this will be tolerance or 0
        delta_pixels: If record_delta, a list of the offsets of every changed pixel, changed_pixel_count in length
    """

    delta_pixels = None
    if record_delta:
        delta_pixels = []
    s=time.time()
    changed_pixel_count = 0
    pixel_count = RESOLUTION[0] * RESOLUTION[1]
    pixel_step = int(pixel_count / round((sample_percentage/100) * pixel_count))
    logging.debug("pct, step = %f, %d", sample_percentage, pixel_step)
    current_pixels = image_2.reshape(pixel_count, 3)
    prev_pixels = image_1.reshape(pixel_count, 3)
    for pixel_index in range(0, pixel_count, pixel_step):
        if abs(int(current_pixels[pixel_index][1]) - int(prev_pixels[pixel_index][1])) > PIXEL_SHIFT_SENSITIVITY:
            changed_pixel_count += 1
            if record_delta:
                delta_pixels.append(pixel_index)
            if tolerance and changed_pixel_count > tolerance:
              logging.debug("Image diff short circuited at: {}".format(time.time() - s))
              return (changed_pixel_count, delta_pixels)
    logging.debug("Image diff took: {}".format(time.time() - s))
    return (changed_pixel_count, delta_pixels)


class ImageCapture(object):
    def __init__(self, is_camera_pi, key_frame_queue):
        self._stop = False
        self._set_camera_type(is_camera_pi)
        self._key_frame_queue = key_frame_queue
        self._last_frame_at = 0.0
        self._frame_delay_secs = 1.0/CAPTURE_RATE_FPS
        self._current_frame_seq = 0

    def stop(self):
        logging.debug("stop()")
        self._stop = True

    def _set_camera_type(self, use_pi_camera):
        if use_pi_camera:
            self._frame_provider = PiCamera()
        else:
            self._frame_provider = WebCamera()

    def get_next_frame(self):
        delay = (self._last_frame_at + self._frame_delay_secs) - time.time()
        if delay > 0:
            time.sleep(delay)
        self._current_frame = self._frame_provider.get_frame()
        self._last_frame_at = time.time()
        self._current_frame_seq += 1

    def is_image_delta_over_threshold(self, changed_pixels_threshold):
        changed_pixels, _ = calculate_image_delta(self._prev_frame, self._current_frame, tolerance=changed_pixels_threshold)
        return changed_pixels > changed_pixels_threshold

    def _train_motion(self):
        "Set the motion threshold to the lowest number of changed pixels that are observed."

        logging.debug("Training motion")
        trained = False
        try:
            self._motion_threshold = 9999
            self.get_next_frame()
            for i in range(TRAINING_SAMPLES):
                self._prev_frame = self._frame_provider.get_frame()
                self.get_next_frame()
                motion, _ = calculate_image_delta(self._prev_frame, self._current_frame)
                self._motion_threshold = min(motion, self._motion_threshold)
            trained = True
        except Exception as e:
            logging.exception("Error training motion")
        logging.debug("Trained {}".format(trained))
        return trained

    def configure_capture(self):
        self._frame_provider._init_camera()
        if not self._train_motion():
            logging.error("Unable to train motion, exiting.")
            return False
        return True

    def capture_frames(self):
        self._current_frame_seq = 0
        logging.info("capturing frames")
        self.get_next_frame()  # To give the initial motion detection a baseline
        while not self._stop:
           self._prev_frame = self._current_frame
           self.get_next_frame()
           if self.is_image_delta_over_threshold(self._motion_threshold):
               logging.debug("Motion detected")
               self._key_frame_queue.put((self._current_frame_seq, self._current_frame))
        logging.info("captured %d frames", self._current_frame_seq)

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
    "Generate an image with the delta between the frames hilighted by dimming all other pixels."

    logging.info("frames %d,%d", frames[0][0], frames[1][0])
    _, delta_pixels = calculate_image_delta(frames[0][1], frames[1][1], sample_percentage=100, record_delta=True)
    total_delta = len(delta_pixels)
    logging.debug("%d pixels changed", total_delta)
    if not total_delta:
        return (0, None)
    delta_index = 0
    highlight_image = frames[1][1].copy()
    delta_completed = False
    dim_by = float(PIXEL_DIMMING_PERCENTAGE)/100
    for pixel_x in range(0, RESOLUTION[1]):
        for pixel_y in range(0, RESOLUTION[0]):
            pixel_index = pixel_x*RESOLUTION[0]+pixel_y
            if delta_completed or pixel_index < delta_pixels[delta_index]:
                highlight_image[pixel_x][pixel_y][:] = [x * dim_by for x in highlight_image[pixel_x][pixel_y]]
            else:
                delta_index += 1
                if delta_index == len(delta_pixels):
                    delta_completed = True
                    break
        if delta_completed:
            break
    return (total_delta, highlight_image)


def display_key_frame_pairs(key_frames, new_frame):
    "Collect pairs of new_frame and generate and display deltas of every pair."

    key_frames.append(new_frame)
    if len(key_frames) == 2:
        total_delta, delta_image = generate_delta_image(key_frames)
        if total_delta:
            cv2.imshow("frame {} - {}".format(key_frames[1][0], total_delta), delta_image)
            cv2.waitKey(50)
        key_frames = [key_frames[1]]
    return key_frames


def main():
    logging.info("running %s", sys.argv[0])
    key_frame_queue = queue.Queue()
    frame_capturer = ImageCapture(_Pi, key_frame_queue)
    frame_capturer.configure_capture()
    frame_source = threading.Thread(target=frame_capturer.capture_frames)
    # start frame provider and give the capture thread enough time to capture frames
    frame_source.start()
    time.sleep(DEMO_CAPTURE_SECS)
    frame_capturer.stop()
    frame_source.join()
    key_frames = []
    num_key_pairs = 0
    # collect all of the frames with motion that were detected
    while not key_frame_queue.empty():
        if len(key_frames) == 1:
            num_key_pairs += 1
        key_frames = display_key_frame_pairs(key_frames, key_frame_queue.get())
        
    logging.info("displayed %d key frames in %f seconds", num_key_pairs, DEMO_CAPTURE_SECS)
    logging.info("waiting for keypress")
    cv2.waitKey()
    cv2.destroyAllWindows()
    logging.info("exiting %s", sys.argv[0])
    sys.exit(0)


if __name__ == "__main__":
    main()

from PIL import Image, ImageDraw
from google.cloud.vision.likelihood import Likelihood
import io
import os
import sys

# first: gcloud auth application-default login

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a client
vision_client = vision.Client()

# The name of the image file to annotate
file_name = os.path.join(
    os.path.dirname(__file__),
    sys.argv[1])

with io.open(file_name, 'rb') as image_file:
     content = image_file.read()

# Loads the image into memory
image = vision_client.image(content=content)

faces = image.detect_faces(limit=30)
print "%d faces" % len(faces)
im = Image.open(io.BytesIO(content))
draw = ImageDraw.Draw(im) 
frame_color_joy = (128,256,128)
frame_color_angry = (236,128,128)
frame_color_meh = (192,192,256)
angry_faces = 0
joyful_faces = 0
for this_face in faces:
   frame_color = frame_color_meh
   print "Confidence %f" % this_face.detection_confidence
   if this_face.joy is Likelihood.VERY_LIKELY or this_face.joy is Likelihood.LIKELY:
       print "joy"
       frame_color = frame_color_joy
       joyful_faces += 1
   if this_face.anger is Likelihood.VERY_LIKELY or this_face.anger is Likelihood.LIKELY:
       print "anger"
       frame_color = frame_color_angry
       angry_faces += 1
   first = this_face.bounds.vertices[0]
   start = first
   for end in this_face.bounds.vertices[1:]:
       print "(%d,%d - %d,%d)" % (start.x_coordinate, start.y_coordinate, end.x_coordinate, end.y_coordinate)
       draw.line((start.x_coordinate,start.y_coordinate, end.x_coordinate, end.y_coordinate), fill=frame_color, width=2)
       start = end
   draw.line((start.x_coordinate,start.y_coordinate, first.x_coordinate, first.y_coordinate), fill=frame_color, width=2)
im.show()
if joyful_faces and joyful_faces > angry_faces:
   print "YAY!!! A happy scene!"
if angry_faces and joyful_faces < angry_faces:
   print "Uh oh!!!  An angry scene!"

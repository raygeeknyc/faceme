#!/usr/bin/python3
import logging
logging.getLogger().setLevel(logging.INFO)

# Import the packages we need for drawing and displaying images
from PIL import Image, ImageDraw, ImageTk
import tkinter

# Imports the Google Cloud client packages we need
from google.cloud import vision
from google.cloud.vision import types

# Import the packages we need for reading parameters and files
import os
import io
import sys

# first you have to authenticate for the default application:
#   gcloud auth application-default login

# Instantiates a vision service client
client = vision.ImageAnnotatorClient()

# Enumerate the likelihood names that are defined by Cloud Vision 1
likelihood_names = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
  'LIKELY', 'VERY_LIKELY')

image_window = None

def showCurrentImage(window, image, window_title):
  if window == None:
    window = tkinter.Tk()

  width, height = image.size
  if width > window.winfo_screenwidth() or height > window.winfo_screenheight():
    resize_ratio = min(float(window.winfo_screenwidth())/width, float(window.winfo_screenheight())/height)
    new_size = int(float(width)*resize_ratio), int(float(height)*resize_ratio)
    print("Resizing large image to: {}".format(new_size))
    image = image.resize(new_size, Image.ANTIALIAS)
  window.title(window_title)
  canvas = tkinter.Canvas(window, width = width, height = height)
  canvas.pack()

  photo = ImageTk.PhotoImage(image=image)
  canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)
  window.mainloop()

def loadImageFile(filename):
  # Loads the image into memory
  # Return the image content
  with io.open(filename, 'rb') as image_file:
    content = image_file.read()
  return content

def setImage(rawContent):
  # Send the image to the cloud vision service to a analyze
  # Return the Google Vision Image
  image = types.Image(content=rawContent)
  return image

def findFaces(image, canvas):
  # Tell the vision service to look for faces in the image
  faces = client.face_detection(
    image = image, image_context = None, max_results = 64).face_annotations
  print("{} faces".format(len(faces)))

  frame_color_joy = (64,256,128)
  frame_color_angry = (256,64,128)
  frame_color_meh = (128,128,256)
  angry_faces = 0
  joyful_faces = 0
  for this_face in faces:
    if this_face.detection_confidence < 0.9:
      frame_width = 1
    else:
      frame_width = 4
      # Classify this face as joyful, angry or meh
      frame_color = frame_color_meh
      if likelihood_names[this_face.joy_likelihood] is 'VERY_LIKELY' or likelihood_names[this_face.joy_likelihood] is 'LIKELY':
        print("joy")
        frame_color = frame_color_joy
        joyful_faces += 1
      if likelihood_names[this_face.anger_likelihood] is 'VERY_LIKELY' or likelihood_names[this_face.anger_likelihood] is 'LIKELY':
        print("anger")
        frame_color = frame_color_angry
        angry_faces += 1
      # Draw a frame around this face's bounding polygon
      first = this_face.bounding_poly.vertices[0]
      start = first
      for end in this_face.bounding_poly.vertices[1:]:
        canvas.line((start.x, start.y, end.x, end.y), fill=frame_color, width=frame_width)
        start = end
      canvas.line((start.x, start.y, first.x, first.y), fill=frame_color, width=frame_width)
  return (len(faces), joyful_faces, angry_faces)

def main(image_filenames):
  for image_filename in image_filenames:
    content = loadImageFile(image_filename)
    image = setImage(content)

    # Create a PIL image that we can draw on
    im = Image.open(io.BytesIO(content))
    canvas = ImageDraw.Draw(im) 

    # Show all labels detected for the image
    labels = client.label_detection(image=image).label_annotations
    print("Labels:")
    for label in labels:
      print(label.description)

    # Show all of the faces detected for the image
    (faces, joyful_faces, angry_faces) = findFaces(image, canvas)
    if faces:
      # Show the image with highlights and report on the majority of moods detected
      if joyful_faces and joyful_faces > angry_faces:
        print("YAY!!! A happy scene!")
      elif angry_faces and joyful_faces < angry_faces:
        print("Uh oh!!!  An angry scene!")
      elif joyful_faces or angry_faces:
        print("A mixed crowd.")
      else:
        print("meh.")
    else:
      print("No faces")
    showCurrentImage(image_window, im, os.path.basename(image_filename))

if __name__ == "__main__":
  if len(sys.argv) < 2:
    logging.error("USAGE: %s img_filename1 [image_filename2]...", sys.argv[0])
    sys.exit(-1)
  else:
    logging.info("%d files.", len(sys.argv)-1)
    main(sys.argv[1:])

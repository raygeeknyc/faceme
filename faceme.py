from PIL import Image, ImageDraw
from google.cloud.vision.likelihood import Likelihood
import io
import os
import sys

# first you have to authenticate for the default application: gcloud auth application-default login

# Imports the Google Cloud client library
from google.cloud import vision

# Instantiates a vision service client
vision_client = vision.Client()


def loadImageFile(filename):
# Loads the image into memory
# Return the image way content
    file_name = os.path.join(
        os.path.dirname(__file__),
        filename)
    with io.open(file_name, 'rb') as image_file:
        content = image_file.read()
    return content
        

def setImage(rawContent):
# Send the image to the cloud vision service to a analyze
# Return the Google Vision Image
    image = vision_client.image(content=rawContent)
    return image

def findFaces(image, canvas):
    # Tell the vision service to look for faces in the image
    faces = image.detect_faces(limit=30)
    print "%d faces" % len(faces)

    frame_color_joy = (128,256,128)
    frame_color_angry = (236,128,128)
    frame_color_meh = (192,192,256)
    angry_faces = 0
    joyful_faces = 0
    for this_face in faces:
       print "Confidence %f" % this_face.detection_confidence
       # Classify this face as joyful, angry or meh
       frame_color = frame_color_meh
       if this_face.joy is Likelihood.VERY_LIKELY or this_face.joy is Likelihood.LIKELY:
           print "joy"
           frame_color = frame_color_joy
           joyful_faces += 1
       if this_face.anger is Likelihood.VERY_LIKELY or this_face.anger is Likelihood.LIKELY:
           print "anger"
           frame_color = frame_color_angry
           angry_faces += 1
       # Draw a frame around this face's bounding polygon
       first = this_face.bounds.vertices[0]
       start = first
       for end in this_face.bounds.vertices[1:]:
           canvas.line((start.x_coordinate,start.y_coordinate, end.x_coordinate, end.y_coordinate), fill=frame_color, width=2)
           start = end
       canvas.line((start.x_coordinate,start.y_coordinate, first.x_coordinate, first.y_coordinate), fill=frame_color, width=2)
    return (len(faces), joyful_faces, angry_faces)

# Process the filenames specified on the command line
for image_filename in sys.argv[1:]:
    content = loadImageFile(image_filename)
    image = setImage(content)

    # Performs label detection on the image file
    labels = image.detect_labels()
    # Create a PIL image that we can draw on
    im = Image.open(io.BytesIO(content))
    canvas = ImageDraw.Draw(im) 
    print('Labels:')
    for label in labels:
        print(label.description)

    descriptions = [label.description for label in labels]
    (faces, joyful_faces, angry_faces) = findFaces(image, canvas)
    if faces:
        # Show the image with highlights and report on the majority of moods detected
        if joyful_faces and joyful_faces > angry_faces:
            print "YAY!!! A happy scene!"
        elif angry_faces and joyful_faces < angry_faces:
            print "Uh oh!!!  An angry scene!"
        elif joyful_faces or angry_faces:
            print "A mixed crowd."
        else:
            print "meh."
    else:
        print 'No faces'
    im.show()

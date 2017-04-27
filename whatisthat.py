# Import the packages we need for reading args and files
import io
import sys

# Import the Google Cloud client library
from google.cloud import vision

# first you have to authenticate for the default application: gcloud auth application-default login

# Instantiates a vision service client
vision_client = vision.Client()

# Load the image into memory from the file named by the first parameter
with io.open(sys.argv[1], 'rb') as image_file:
    content = image_file.read()
    image = vision_client.image(
        content=content)

# Perform label detection on the image file
labels = image.detect_labels()
print('Labels:')
for label in labels:
    print(label.description)

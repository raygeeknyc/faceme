import logging
logging.getLogger().setLevel(logging.DEBUG)

import io
import sys

# Import the packages we need for drawing and displaying images
from PIL import Image
import numpy
import cv2

RANGE=1

def get_avg_pixel(point, pixels):
    "Return an average RGB value for the neighborhood of the point in image."

    neighborhood_area=(RANGE*2+1)**2
    pixel_count = 0
    average_pixel=[0,0,0]
    for neighbor_x in range(point[0]-RANGE, point[0]+RANGE+1):
        for neighbor_y in range(point[1]-RANGE, point[1]+RANGE+1):
            pixel_count += 1
            average_pixel = [average_pixel[channel]+pixels[neighbor_x, neighbor_y][channel] for channel in range(len(average_pixel))]
    average_pixel=[int(average_pixel[channel]/neighborhood_area) for channel in range(len(average_pixel))]
    return tuple(average_pixel)

def blur_image_region(image, region_box):
    """
    Blur each pixel in region_box of image.
    Args:
        image: an image with [x][y] pixels
        region_box: two points (top_left, bottom_right), each (x, y)
    Returns:
        region_area: The number of pixels blurred
    """
    
    pixels = image.load()
    image_size = image.size
    for region_x in range(max(RANGE, region_box[0][0]), min(image_size[0]-(RANGE+1), region_box[1][0])):
        for region_y in range(max(RANGE, region_box[0][1]), min(image_size[1]-(RANGE+1), region_box[1][1])):
            point = (region_x, region_y)
            blurred_pixel = get_avg_pixel(point, pixels)
            pixels[region_x, region_y] = blurred_pixel
    blurred_area = ( 
        (min(image_size[0]-(RANGE+1), region_box[1][0]) - max(RANGE, region_box[0][0])) *
        (min(image_size[1]-(RANGE+1), region_box[1][1]) - max(RANGE, region_box[0][1]))
        )
    logging.debug("blurred %d pixels", blurred_area)
    return blurred_area


def main(image_filename):
    with io.open(image_filename, 'rb') as image_file:
        content = image_file.read()
    image = Image.open(io.BytesIO(content))
    pixels = image.load()
    print("{}:{}".format(pixels, image.size))
    # define a region of the top half of the image
    blur_region=[(0, 0), (image.size[0], int(image.size[1]/2))]
    blurred_area = blur_image_region(image, blur_region)
    cv2.imshow("blurred {}".format((blur_region, blurred_area)), numpy.array(image)[:, :, ::-1].copy())
    cv2.waitKey(0)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        logging.error("usage %s image_filename", sys.argv[0])
        sys.exit(-1)
    main(sys.argv[1])
    sys.exit(0)

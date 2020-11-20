import cv2
import numpy as np
import sys

def main(filename):
  img = cv2.imread(filename)
  height, width, _ = img.shape
  mask = np.zeros(img.shape[:2],np.uint8)

  bgdModel = np.zeros((1,65),np.float64)
  fgdModel = np.zeros((1,65),np.float64)

  rect = (5,5,width-6,height-6)
  cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

  mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
  foreground_img = img*mask2[:,:,np.newaxis]

  cv2.imshow("image", np.array(img)[:, :, ::-1].copy())
  cv2.imshow("foreground", np.array(foreground_img)[:, :, ::-1].copy())
  cv2.waitKey(0)

if len(sys.argv) < 2:
  sys.stderr.write("{} <image-filename>\n".format(sys.argv[0]))
  sys.exit(-1)
main(sys.argv[1])

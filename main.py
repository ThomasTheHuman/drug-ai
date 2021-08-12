import sys
from PIL import Image
sys.modules['Image'] = Image
from picamera import PiCamera
from time import sleep
import loader
from cnn import DrugAI

batch_size = 16
img_size = 180

if __name__ == '__main__':
    ai = DrugAI(10, img_size, 'checkpoint.ckpt')
    ai.compile()
    
    with PiCamera() as cam:
        while 1:
            cam.start_preview()
            sleep(5)
            cam.capture('tmp.jpg')
            cam.stop_preview()
            ai.predict('tmp.jpg')

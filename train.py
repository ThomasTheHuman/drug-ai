import sys
from PIL import Image
sys.modules['Image'] = Image
from picamera import PiCamera
from time import sleep
import loader
from drugai import DrugAI
import config

if __name__ == '__main__':
    training_ds, validation_ds = loader.load_dataset(img_size, batch_size)
    ai = DrugAI(10, img_size, 'checkpoint.ckpt')
    ai.compile()
    ai.train(training_ds, validation_ds)
    ai.convert_to_lite()

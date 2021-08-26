import sys
from PIL import Image
sys.modules['Image'] = Image
from picamera import PiCamera
from time import sleep
import loader
from drugai import DrugAI
from config import img_size
from config import class_names
from config import batch_size
from config import checkpoint

if __name__ == '__main__':
    training_ds, validation_ds = loader.load_dataset(img_size, batch_size)
    ai = DrugAI(len(class_names), img_size, checkpoint, class_names)
    ai.compile()
    ai.train(training_ds, validation_ds)
    ai.convert_to_lite()

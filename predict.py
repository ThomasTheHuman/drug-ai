import tensorflow as tf
from tensorflow import keras
import numpy as np
from picamera import PiCamera
from time import sleep
from config import img_size
from config import class_names

def predict(file_name, _interpreter):
    img = keras.preprocessing.image.load_img(
        file_name, target_size=(img_size, img_size)
    )
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    i = _interpreter.get_input_details()[0]
    o = _interpreter.get_output_details()[0]
    _interpreter.set_tensor(i['index'], img_array)
    _interpreter.invoke()
    prediction = _interpreter.get_tensor(o['index'])[0]
    score = tf.nn.softmax(prediction)
    print("This image is {} ({}% confidence)".format(class_names[np.argmax(score)], 100 * np.max(score)), end='\r')


if __name__ == '__main__':
    interpreter = tf.lite.Interpreter('model.tflite')
    interpreter.allocate_tensors()
    with PiCamera() as cam:
        while 1:
            cam.start_preview()
            # sleep(5)
            cam.capture('tmp.jpg')
            cam.stop_preview()
            predict('tmp.jpg', interpreter)



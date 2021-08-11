import tensorflow as tf
import pathlib

# loads dataset
def load_dataset(img_size, batch_size):
    dataset_folder = pathlib.Path('dataset')
    # uses 80% of images for training
    training = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_folder,
        validation_split=0.2,
        subset="training",
        seed=1,
        image_size=(img_size, img_size),
        batch_size=batch_size)
    # 20% of images are used for validation
    validation = tf.keras.preprocessing.image_dataset_from_directory(
        dataset_folder,
        validation_split=0.2,
        subset="validation",
        seed=1,
        image_size=(img_size, img_size),
        batch_size=batch_size)
    # training dataset is shuffled, both datasets are autotuned and returned as tuple
    return \
        training.cache().shuffle(100).prefetch(buffer_size=tf.data.AUTOTUNE), \
        validation.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

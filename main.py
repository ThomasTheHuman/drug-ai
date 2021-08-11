import loader
from cnn import DrugAI

batch_size = 16
img_size = 180

if __name__ == '__main__':
    training_ds, validation_ds = loader.load_dataset(img_size, batch_size)
    ai = DrugAI(10, img_size, 'checkpoint.ckpt')
    ai.compile()
    # uncomment in order to train network
    # ai.train(training_ds, validation_ds)
    ai.predict('test_img2.png')
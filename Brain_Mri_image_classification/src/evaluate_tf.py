import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def evaluate(model_path, test_dir):
    model = tf.keras.models.load_model(model_path)
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_gen = test_datagen.flow_from_directory(test_dir, target_size=(224,224), color_mode='grayscale',
                                                batch_size=8, class_mode='sparse', shuffle=False)
    loss, acc = model.evaluate(test_gen)
    print(f"Test loss: {loss:.4f}, Test acc: {acc:.4f}")

if __name__ == '__main__':
    evaluate('experiments/tf_model.h5','data/raw/val')

\
import argparse, os
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def build_model(input_shape=(224,224,1), num_classes=2):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizers.Adam(1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def main(args):
    train_dir = args.train_dir
    val_dir = args.val_dir
    batch_size = args.batch_size
    epochs = args.epochs

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.05,
                                       height_shift_range=0.05, horizontal_flip=True)
    val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(train_dir, target_size=(224,224), color_mode='grayscale',
                                                  batch_size=batch_size, class_mode='sparse')
    val_gen = val_datagen.flow_from_directory(val_dir, target_size=(224,224), color_mode='grayscale',
                                                  batch_size=batch_size, class_mode='sparse')

    model = build_model(input_shape=(224,224,1), num_classes=len(train_gen.class_indices))
    model.summary()

    history = model.fit(train_gen, validation_data=val_gen, epochs=epochs)
    out = os.path.join('experiments','tf_model.h5')
    model.save(out)
    print("Saved model to", out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', default='data/raw/train')
    parser.add_argument('--val_dir', default='data/raw/val')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()
    main(args)

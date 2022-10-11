from statistics import mode
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tflite_runtime.interpreter as tflite
import pathlib

def load_data(path):
    batch_size = 32
    img_height = 180
    img_width = 180
    data_dir = pathlib.Path(path)


    train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

    return train_ds, val_ds

def create_model(train_ds, val_ds):
    num_classes = 6

    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(),  # input layer (1)
        tf.keras.layers.Dense(128, activation='relu'),  # hidden layer (2)
        tf.keras.layers.Dense(num_classes, activation='softmax') # output layer (3)
    ])

    model.compile(
    optimizer='adam',
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

    model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1
    )

    model.save('my_model')
    return model

def load_model(path):
    new_model = tf.keras.models.load_model(path)
    return new_model

def predict(model, train_ds):
    for images, labels in train_ds.take(10):
        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
        plt.show()
        predictions = model.predict(images)
        print(np.argmax(predictions[0]))


def main():
    train_ds, val_ds = load_data("train");

    choice = input("1. load 2. create: ")

    if choice == "1":
        model = tf.keras.models.load_model('my_model')
    else :
        model = create_model(train_ds, val_ds)

    predict(model, train_ds)

if __name__ == "__main__":
    main()

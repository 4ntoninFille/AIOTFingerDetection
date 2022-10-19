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

    train_ds = train_ds.shuffle(3)

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

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('my_model.tflite', 'wb') as f:
        f.write(tflite_model)
    
    return model

def load_model(path):
    new_model = tf.keras.models.load_model(path)
    return new_model

def load_model_lit(path):
    interpreter = tf.lite.Interpreter(path)
    interpreter.allocate_tensors()
    return interpreter



def predict(interpreter, train_ds):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    
    for images, labels in train_ds.take(3):

        image = np.expand_dims(images[0], axis=0)

        plt.imshow(images[0].numpy().astype("uint8"))
        plt.axis("off")
        plt.show()
        interpreter.set_tensor(input_details[0]['index'], image)

        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        print('number of fingers:{}'.format(np.argmax(output_data)))

def main():
    train_ds, val_ds = load_data("train");

    choice = input("\n\n1. load\t\t2. create\n:")

    if choice == "1":
        # load_model('my_model')
        interpreter = load_model_lit("my_model.tflite")
    else :
        model = create_model(train_ds, val_ds)
        interpreter = load_model_lit("my_model.tflite")

    predict(interpreter, train_ds)

if __name__ == "__main__":
    main()

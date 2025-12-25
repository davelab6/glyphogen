#!/usr/bin/env python
import tensorflow as tf
import keras
from keras import layers
import os

from deepvecfont3.model import VectorizationGenerator
from deepvecfont3.hyperparameters import NUM_GLYPHS, EPOCHS, GEN_IMAGE_SIZE

def get_data():
    # Create real dataset
    if not os.path.exists("train.tfds"):
        train_dataset, test_dataset = create_real_dataset()
        train_dataset.save("train.tfds")
        test_dataset.save("test.tfds")

    # Load 'em anyway, because using an on-disk version saves memory
    train_dataset = tf.data.Dataset.load("train.tfds")
    test_dataset = tf.data.Dataset.load("test.tfds")
    return train_dataset, test_dataset


def build_debugging_model():
    # Take the encoder from the VectorizationGenerator
    vectorizer = VectorizationGenerator(
        num_transformer_layers=1, d_model=1, num_heads=1, dff=1
    )
    
    input_shape = (GEN_IMAGE_SIZE[0], GEN_IMAGE_SIZE[1], 1)
    input_layer = keras.Input(shape=input_shape)

    x = vectorizer.conv1(input_layer)
    x = vectorizer.norm1(x)
    x = vectorizer.relu1(x)
    x = vectorizer.conv2(x)
    x = vectorizer.norm2(x)
    x = vectorizer.relu2(x)
    x = vectorizer.conv3(x)
    x = vectorizer.norm3(x)
    x = vectorizer.relu3(x)
    x = vectorizer.conv4(x)
    x = vectorizer.norm4(x)
    x = vectorizer.relu4(x)
    x = vectorizer.conv5(x)
    x = vectorizer.norm5(x)
    x = vectorizer.relu5(x)
    x = vectorizer.flatten(x)
    x = vectorizer.dense(x)
    x = vectorizer.norm_dense(x)
    x = vectorizer.sigmoid(x)
    encoder_output = vectorizer.output_dense(x)

    output_layer = layers.Dense(NUM_GLYPHS, activation="softmax")(encoder_output)

    model = keras.Model(inputs=input_layer, outputs=output_layer)
    return model


def main():
    train_dataset, test_dataset = get_data()

    # Adapt the dataset for the debugging task
    def to_classification_dataset(x, y):
        return y["raster"], x[1]

    train_dataset = train_dataset.map(to_classification_dataset)
    test_dataset = test_dataset.map(to_classification_dataset)

    model = build_debugging_model()

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Training the debugging model...")
    model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=test_dataset,
    )


if __name__ == "__main__":
    main()

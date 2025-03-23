# model_factory.py

import tensorflow as tf

INPUT_SHAPE = (128, 130, 1)
NUM_CLASSES = 4

def get_inception_model():
    base_model = tf.keras.applications.InceptionV3(
        include_top=False,
        weights=None,
        input_shape=INPUT_SHAPE,
        pooling='max'
    )

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_resnet_model():
    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=INPUT_SHAPE,
        pooling='max'
    )

    model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def get_custom_cnn_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=INPUT_SHAPE),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# Optional: wrapper
def get_model(model_name: str):
    if model_name.lower() == 'inception':
        return get_inception_model()
    elif model_name.lower() == 'resnet':
        return get_resnet_model()
    elif model_name.lower() == 'custom':
        return get_custom_cnn_model()
    else:
        raise ValueError("Model name must be 'inception', 'resnet', or 'custom'")

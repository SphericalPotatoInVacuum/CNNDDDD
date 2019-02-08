import tensorflow as tf

img_rows = 28
img_cols = 28
num_classes = 10

mnist, x_train, x_test, y_train, y_test = [None] * 5


def load_data():
    global mnist, x_train, x_test, y_train, y_test

    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = [img_rows, img_cols, 1]

    x_train = x_train / 255.0
    x_test = x_test / 255.0


def get_data():
    return (x_train, y_train), (x_test, y_test)


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=32, kernel_size=[5, 5], activation=tf.nn.relu,
                               input_shape=[28, 28, 1]),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
        tf.keras.layers.Conv2D(filters=64, kernel_size=[5, 5], activation=tf.nn.relu),
        tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(1024, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

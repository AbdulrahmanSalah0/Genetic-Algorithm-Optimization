import sys
import tensorflow as tf

# Use MNIST handwriting dataset (which contains people handwritten digits and its a builtin data set which is builtin to the library.)
mnist = tf.keras.datasets.mnist

# Prepare data for training
# By turning the data into a format that i ca put into my convolutional neural network

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Taking all the values and divide them by 255 to put them in 0 to 1 range to be a little bit easier to train on.
x_train, x_test = x_train / 255.0, x_test / 255.0
# and then categorizing the data to be in a nice usable format.
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
x_train = x_train.reshape(
    x_train.shape[0], x_train.shape[1], x_train.shape[2], 1
)
x_test = x_test.reshape(
    x_test.shape[0], x_test.shape[1], x_test.shape[2], 1
)
# Create a convolutional neural network using a sequential model.
model = tf.keras.models.sequantial([

    # Convolutional layer. Learn 32 filters using a 3x3 kernel matrix
    tf.keras.layers.Conv2D(
        # A rectified linear unit (ReLU) is an activation function that introduces the property of
        # non-linearity to a deep learning model and solves the vanishing gradients issue
        32, (3, 3), activation="relu", input_shape=(28, 28, 1)
    ),

    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten units
    tf.keras.layers.Flatten(),

    # Adding our hidden layers
    tf.keras.layers.Dense(128, activation="relu"),
    # Adding Dropout to prevent over-fitting and over-lying on any particular node and generalize
    tf.keras.layers.Dropout(0.5),

    # Add an output layer with an output units for all 10 digits from 0 to 9
    tf.keras.layers.Dense(10, activation="softmax")
    # softmax activation function: it will take our output and turn it into
    # a probability distribution formula
])

# Train neural network
model.py_compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
# Fitting the model to all my training data
model.fit(x_train, y_train, epochs=10)

# Evaluating to see neural network performance
model.evaluate(x_test, y_test, verbose=2)

# Saving model to file
if len(sys.argv) ==2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")
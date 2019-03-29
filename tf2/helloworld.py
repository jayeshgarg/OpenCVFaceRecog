import tensorflow as tf

mnist = tf.keras.datasets.mnist

# load numpy bases test data into arrays
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# reduce data size for demo purpose and faster computation
x_train, x_test = x_train / 255.0, x_test / 255.0

# what is a shape? how to visualize shape for better understanding?
# what is sequential model?
model = tf.keras.models.Sequential([
    # what is flatten in layers?
    # what does input shape explain here?
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # what is dense here? and what does activation mechanism 'relu' means?
    tf.keras.layers.Dense(128, activation='relu'),
    # what is dropout rate? does it means that 0.2 (20%) of data will be randomly erased to create data set?
    tf.keras.layers.Dropout(0.2),
    # what does activation mechanism 'softmax' means?
    tf.keras.layers.Dense(10, activation='softmax')
])

# what is an optimizer and how many are there? what does adam optimizer do?
model.compile(optimizer='adam',
              # what is loss? what does sparse_categorical_crossentropy means?
              loss='sparse_categorical_crossentropy',
              # what different metrics are available?
              metrics=['accuracy'])

# this is the command that needs to be called to train the model
# epoch is the number of iterations that will be used to train the model. the training is cumulative in nature.
# so more epoch, better accuracy, but lot more time as well
model.fit(x_train, y_train, epochs=50)

# this command takes care of the evaluation of actual data over trained model
model.evaluate(x_test, y_test)

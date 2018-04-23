import tensorflow as tf


def build_model(input_size, hidden_layers, output_size):
    x = tf.placeholder(shape=[None, input_size], dtype=tf.float32, name='input')
    z = x

    last_size = input_size
    if hidden_layers is not None:
        for layer in hidden_layers:
            W = tf.Variable(tf.random_uniform([last_size, layer], 0, 0.01))
            b = tf.Variable(tf.constant(0.1, shape=[layer]))
            z = tf.nn.relu(tf.matmul(z, W) + b)
            last_size = layer

    W = tf.Variable(tf.random_uniform([last_size, output_size], 0, 0.01))
    b = tf.Variable(tf.constant(0.1, shape=[output_size]))
    y = tf.add(tf.matmul(z, W), b, name='output')
    prediction = tf.argmax(y, 1, name='prediction')

    return x, y, prediction



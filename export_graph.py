# https://github.com/llSourcell/A_Guide_to_Running_Tensorflow_Models_on_Android/blob/master/tensorflow_model/mnist_convnet.py
import tensorflow as tf
import os
from tensorflow.python.tools import freeze_graph
from tensorflow.python.tools import optimize_for_inference_lib

from ModelBuilder import build_model

MODEL_NAME = 'tictactoe'

def export_model(input_node_name, output_node_name):
    freeze_graph.freeze_graph('out/' + MODEL_NAME + '.pbtxt', None, False,
        'out/' + MODEL_NAME + '.chkp', output_node_name, "save/restore_all",
        "save/Const:0", 'out/frozen_' + MODEL_NAME + '.pb', True, "")

    input_graph_def = tf.GraphDef()
    with tf.gfile.Open('out/frozen_' + MODEL_NAME + '.pb', "rb") as f:
        input_graph_def.ParseFromString(f.read())

    output_graph_def = optimize_for_inference_lib.optimize_for_inference(
            input_graph_def, [input_node_name], [output_node_name],
            tf.float32.as_datatype_enum)

    with tf.gfile.FastGFile('out/opt_' + MODEL_NAME + '.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())

    print("graph saved!")


layers = [729, 729, 729]
inputs, Q_out, predict = build_model(9, layers, 9)


sess = tf.Session()


saver = tf.train.Saver()

init = tf.global_variables_initializer()
sess.run(init)

checkpoint = tf.train.get_checkpoint_state("model")
if checkpoint and checkpoint.model_checkpoint_path:
    s = saver.restore(sess,checkpoint.model_checkpoint_path)
    print("Successfully loaded the model:", checkpoint.model_checkpoint_path)
    step = int(os.path.basename(checkpoint.model_checkpoint_path).split('-')[1])
else:
    print("Could not find old network weights")

tf.train.write_graph(sess.graph_def, 'out',
            MODEL_NAME + '.pbtxt', True)
saver.save(sess, 'out/' + MODEL_NAME + '.chkp')

export_model('input', 'output')

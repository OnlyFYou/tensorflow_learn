# _*_ coding: utf-8 _*_
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import learn.mnist_learn.mnist_inference
import learn.mnist_learn.mnist_train

EVAL_INTERVAL_SECS = 10


def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, learn.mnist_learn.mnist_inference.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, learn.mnist_learn.mnist_inference.OUTPUT_NODE], name='y-input')
        valid_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
        X = mnist.validation.images
        y = learn.mnist_learn.mnist_inference.inference(x, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        variable_averages = tf.train.ExponentialMovingAverage(learn.mnist_learn.mnist_train.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(learn.mnist_learn.mnist_train.MODEL_SAVE_PATH)
                print(ckpt.model_checkpoint_path)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=valid_feed)
                    print('After %s training step validataion accuary = %g' % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):
    mnist = input_data.read_data_sets('/path/to/MNIST_data', one_hot=True)
    evaluate(mnist)

    
if __name__ == '__main__':
    main()

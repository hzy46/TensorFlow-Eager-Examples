import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.examples.tutorials.mnist import input_data
tfe.enable_eager_execution()


W = tf.get_variable(name="W", shape=(784, 10))
b = tf.get_variable(name="b", shape=(10, ))


def softmax_model(image_batch):
    model_output = tf.nn.softmax(tf.matmul(image_batch, W) + b)
    return model_output


def cross_entropy(model_output, label_batch):
    loss = tf.reduce_mean(
        -tf.reduce_sum(label_batch * tf.log(model_output),
        reduction_indices=[1]))
    return loss


@tfe.implicit_value_and_gradients
def cal_gradient(image_batch, label_batch):
    return cross_entropy(softmax_model(image_batch), label_batch)


if __name__ == '__main__':
    data = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
    train_ds = tf.data.Dataset.from_tensor_slices((data.train.images, data.train.labels))\
        .map(lambda x, y: (x, tf.cast(y, tf.float32)))\
        .shuffle(buffer_size=1000)\
        .batch(100)\

    optimizer = tf.train.GradientDescentOptimizer(0.5)

    for step, (image_batch, label_batch) in enumerate(tfe.Iterator(train_ds)):
        loss, grads_and_vars = cal_gradient(image_batch, label_batch)
        optimizer.apply_gradients(grads_and_vars)
        print("step: {}  loss: {}".format(step, loss.numpy()))

    model_test_output = softmax_model(data.test.images)
    model_test_label = data.test.labels
    correct_prediction = tf.equal(tf.argmax(model_test_output, 1), tf.argmax(model_test_label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    print("test accuracy = {}".format(accuracy.numpy()))

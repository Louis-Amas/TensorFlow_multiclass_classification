import tensorflow as tf
import numpy as np
import pandas as pd

def get_one_hot(vector, nb_classes):
    return np.eye(nb_classes)[np.array(vector).reshape(-1)]


class DNNMultiClass():

    def __init__(self, size_features, tab_neurons):
        self.w1 = tf.Variable(tf.random_normal([size_features, tab_neurons[0]]))
        self.b1 = tf.Variable(tf.zeros([tab_neurons[0]]))

        self.w2 = tf.Variable(tf.random_normal([tab_neurons[0], tab_neurons[1]]))
        self.b2 = tf.Variable(tf.zeros([tab_neurons[1]]))

        # self.w3 = tf.Variable(tf.random_normal([tab_neurons[1], tab_neurons[2]]))
        # self.b3 = tf.Variable(tf.zeros([tab_neurons[2]]))

    def forward(self, inputs):
        z1 = tf.matmul(tf.cast(inputs, tf.float32), self.w1) + self.b1
        a1 = tf.nn.relu(z1)
        z2 = tf.matmul(z1, self.w2) + self.b2
        # a2 = tf.nn.relu(z2)
        # z3 = tf.matmul(a2, self.w3) + self.b3
        # a3 = tf.nn.relu(z3)
        return z2

    def train(self, epochs, features, targets):
        tf_features = tf.placeholder(tf.float32, shape=[None, features.shape[1]])
        tf_targets = tf.placeholder(tf.float32, shape=[None, targets.shape[1]])

        py = self.forward(features)

        correct_predictions = tf.equal(tf.argmax(py, 1), tf.argmax(targets, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=tf_targets, logits=py))
        optimizer = tf.train.GradientDescentOptimizer(0.001)
        train = optimizer.minimize(cost)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print('Start training')
            for epoch in range(epochs):
                _, co, acc = sess.run((train, cost, accuracy), feed_dict={
                    tf_features: features,
                    tf_targets: targets
                })
                print(co, acc)



if __name__ == '__main__':
    features = np.random.rand(2, 4)
    df = pd.read_csv('mnist_train.csv')
    targets = get_one_hot(df.ix[:,0].values, 10)
    features = np.delete(df.values, 0, axis=1)
    dnn = DNNMultiClass(features.shape[1], [50, 10])
    dnn.train(1000, features, targets)

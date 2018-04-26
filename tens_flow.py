import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def minibatcher(X, y, batch_size, shuffle):
    assert X.shape[0] == y.shape[0]
    n_samples = X.shape[0]

    if shuffle:
        idx = np.random.permutation(n_samples)
    else:
        idx = list(range(n_samples))

    for k in range(int(np.ceil(n_samples/batch_size))):
        from_idx = k * batch_size
        to_idx = (k+1)*batch_size
        yield X[idx[from_idx:to_idx], :, :, :], y[idx[from_idx:to_idx], :]


def fc_no_activation_layer(in_tensors, n_units):
    w = tf.get_variable('fc_W', [in_tensors.get_shape()[1], n_units], tf.float32,
                        tf.contrib.layers.xavier_initializer())
    variable_summaries(w)
    b = tf.get_variable('fc_B', [n_units, ], tf.float32,
                        tf.constant_initializer(0.0))
    variable_summaries(b)
    # preactivate = tf.matmul(in_tensors, w) + b
    # tf.summary.histogram('pre_activations', preactivate)
    return tf.matmul(in_tensors, w) + b


def fc_layer(in_tensors, n_units):
    # activations = tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))
    # tf.summary.histogram('activations', activations)
    return tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))


def maxpool_layer(in_tensors, sampling):
    return tf.nn.max_pool(in_tensors, [1, sampling, sampling, 1],
                          [1, sampling, sampling, 1], 'SAME')


def conv_layer(in_tensors, kernel_size, n_units):
    # weights
    w = tf.get_variable('conv_W', [kernel_size, kernel_size, in_tensors.get_shape()[3], n_units],
                        tf.float32, tf.contrib.layers.xavier_initializer())
    # variable_summaries(w)
    # biases
    b = tf.get_variable('conv_B', [n_units, ], tf.float32, tf.constant_initializer(0.0))
    # variable_summaries(b)
    # return weights + biases
    return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors, w, [1, 1, 1, 1], 'SAME') + b)


def dropout(in_tensors, keep_proba, is_training):
    # tf.summary.scalar('dropout_keep_probability', keep_proba)
    return tf.cond(is_training, lambda: tf.nn.dropout(in_tensors, keep_proba), lambda: in_tensors)


def model(in_tensors, is_training):

    # First layer: 5x5 2d-conv, 32 filters, 2x maxpool, 20% dropout
    with tf.variable_scope('l1'):
        l1 = maxpool_layer(conv_layer(in_tensors, 5, 32), 2)
        l1_out = dropout(l1, 0.8, is_training)

    # Second layer: 5x5 2d-conv, 64 filters, 2x maxpool, 20% dropout
    with tf.variable_scope('l2'):
        l2 = maxpool_layer(conv_layer(l1_out, 5, 64), 2)
        l2_out = dropout(l2, 0.8, is_training)

    with tf.variable_scope('flatten'):
        l2_out_flat = tf.layers.flatten(l2_out)

    # Fully collected layer, 1024 neurons, 40% dropout
    with tf.variable_scope('l3'):
        l3 = fc_layer(l2_out_flat, 1024)
        l3_out = dropout(l3, 0.6, is_training)

    # Output
    with tf.variable_scope('out'):
        out_tensors = fc_no_activation_layer(l3_out, 25)

    return out_tensors


def variable_summaries(var):

    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def train_model(X_train, y_train, X_test, y_test, learning_rate, max_epochs, batch_size, resized_image, n_classes):

    in_X_tensors_batch = tf.placeholder(tf.float32, shape=(None, resized_image[0], resized_image[1], 1))
    in_y_tensors_batch = tf.placeholder(tf.float32, shape=(None, n_classes))
    is_training = tf.placeholder(tf.bool)

    logits = model(in_X_tensors_batch, is_training)
    with tf.name_scope('softmax'):
        out_y_pred = tf.nn.softmax(logits)

    # cost function
    with tf.name_scope('cross_entropy'):
        loss_score = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=in_y_tensors_batch)
        loss = tf.reduce_mean(loss_score)

    # Optimizer
    with tf.name_scope('train'):
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Monitor cost tensor
    tf.summary.scalar('cost', loss)

    summary_op = tf.summary.merge_all()

    with tf.Session() as session:
        # initialize variables to default values
        session.run(tf.global_variables_initializer())
        # Write logs to tensorboard file.
        summary_writer = tf.summary.FileWriter('./logs/1/train', session.graph)

        for epoch in range(max_epochs):
            print("Epoch=", epoch)
            tf_score = []
            for mb in minibatcher(X_train, y_train, batch_size, shuffle=True):
                _, tf_output, summary = session.run([optimizer, loss, summary_op], feed_dict={in_X_tensors_batch: mb[0],
                                                                      in_y_tensors_batch: mb[1],
                                                                      is_training: True})

                summary_writer.add_summary(summary, epoch)
                tf_score.append(tf_output)
            print(' train_loss_score=', np.mean(tf_score))


        print('TEST SET PERFORMANCE')
        y_test_pred, test_loss = session.run([out_y_pred, loss], feed_dict={in_X_tensors_batch: X_test,
                                                                            in_y_tensors_batch: y_test,
                                                                            is_training: False})

        print('Test_loss_score=', test_loss)
        y_test_pred_classified = np.argmax(y_test_pred, axis=1).astype(np.int32)
        y_test_true_classified = np.argmax(y_test, axis=1).astype(np.int32)
        print(classification_report(y_test_true_classified, y_test_pred_classified))
        print(accuracy_score(y_test_true_classified, y_test_pred_classified))
        cfm = confusion_matrix(y_test_true_classified, y_test_pred_classified)

        plt.clf()

        plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.colorbar()
        plt.tight_layout()
        plt.show()

        plt.imshow(np.log2(cfm + 1), interpolation='nearest', cmap=plt.get_cmap('tab20'))
        plt.colorbar()
        plt.tight_layout()
        plt.show()

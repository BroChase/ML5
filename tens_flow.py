import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def minibatcher(X, y, batch_size, shuffle):
    assert X.shape[0] == y.shape[0]
    n_samples = X.shape[0]

    if shuffle:
        # Randomly permute range n_samples
        idx = np.random.permutation(n_samples)
    else:
        idx = list(range(n_samples))
    # take the ceiling of the n_samples/batch_size
    for k in range(int(np.ceil(n_samples/batch_size))):
        from_idx = k * batch_size
        to_idx = (k+1)*batch_size
        yield X[idx[from_idx:to_idx], :, :, :], y[idx[from_idx:to_idx], :]


def fc_no_activation_layer(in_tensors, n_units):
    with tf.variable_scope('w'):
        w = tf.get_variable('fc_W', [in_tensors.get_shape()[1], n_units], tf.float32,
                            tf.contrib.layers.xavier_initializer())
    with tf.variable_scope('b'):
        b = tf.get_variable('fc_B', [n_units, ], tf.float32,
                            tf.constant_initializer(0.0))

    preactivate = tf.matmul(in_tensors, w) + b
    tf.summary.histogram('pre_activations', preactivate)
    return preactivate  # tf.matmul(in_tensors, w) + b


def fc_layer(in_tensors, n_units):
    activations = tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))
    tf.summary.histogram('activations', activations)
    return activations  # tf.nn.leaky_relu(fc_no_activation_layer(in_tensors, n_units))


def maxpool_layer(in_tensors, sampling):
    return tf.nn.max_pool(in_tensors, [1, sampling, sampling, 1],
                          [1, sampling, sampling, 1], 'SAME')


def conv_layer(in_tensors, kernel_size, n_units):
    # weights
    with tf.variable_scope('w'):
        w = tf.get_variable('conv_W', [kernel_size, kernel_size, in_tensors.get_shape()[3], n_units],
                            tf.float32, tf.contrib.layers.xavier_initializer())
    # biases
    with tf.variable_scope('b'):
        b = tf.get_variable('conv_B', [n_units, ], tf.float32, tf.constant_initializer(0.0))

    # return weights + biases
    return tf.nn.leaky_relu(tf.nn.conv2d(in_tensors, w, [1, 1, 1, 1], 'SAME') + b)


def dropout(in_tensors, keep_proba, is_training):
    # tf.summary.scalar('dropout_keep_probability', keep_proba)
    return tf.cond(is_training, lambda: tf.nn.dropout(in_tensors, keep_proba), lambda: in_tensors)


def model(in_tensors, is_training):

    # First layer: 5x5 2d-conv, 32 filters, 2x maxpool, 10% dropout
    with tf.variable_scope('layer_1'):
        l1 = maxpool_layer(conv_layer(in_tensors, 5, 32), 2)
        l1_out = dropout(l1, 0.9, is_training)

    # Second layer: 5x5 2d-conv, 48 filters, 2x maxpool, 20% dropout
    with tf.variable_scope('layer_2'):
        l2 = maxpool_layer(conv_layer(l1_out, 5, 48), 2)
        l2_out = dropout(l2, 0.8, is_training)

    # Second layer: 5x5 2d-conv, 64 filters, 2x maxpool, 20% dropout
    with tf.variable_scope('layer_3'):
        l3 = maxpool_layer(conv_layer(l2_out, 5, 64), 2)
        l3_out = dropout(l3, 0.8, is_training)

    with tf.variable_scope('flatten'):
        l3_out_flat = tf.layers.flatten(l3_out)

    # Fully collected layer, 1024 neurons, 40% dropout
    with tf.variable_scope('layer_4'):
        l4 = fc_layer(l3_out_flat, 1024)
        l4_out = dropout(l4, 0.6, is_training)

    # Output
    with tf.variable_scope('out'):
        out_tensors = fc_no_activation_layer(l4_out, 25)

    return out_tensors


def train_model(X_train, y_train, X_test, y_test, learning_rate, max_epochs, batch_size, resized_image, n_classes, f):

    in_X_tensors_batch = tf.placeholder(tf.float32, shape=(None, resized_image[0], resized_image[1], 1))
    in_y_tensors_batch = tf.placeholder(tf.float32, shape=(None, n_classes))
    is_training = tf.placeholder(tf.bool)

    logits = model(in_X_tensors_batch, is_training)
    with tf.name_scope('softmax'):
        out_y_pred = tf.nn.softmax(logits)

    # cost function
    with tf.name_scope('cross_entropy'):
        loss_score = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=in_y_tensors_batch)
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
        summary_writer = tf.summary.FileWriter('./logs/train', session.graph)

        for epoch in range(max_epochs):
            print("Epoch=", epoch)
            # f.write('Epoch={0:2d}\n'.format(epoch))
            tf_score = []
            for mb in minibatcher(X_train, y_train, batch_size, shuffle=True):
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)

                _, tf_output, summary = session.run([optimizer, loss, summary_op], feed_dict={in_X_tensors_batch: mb[0],
                                                                      in_y_tensors_batch: mb[1],
                                                                      is_training: True}, options=run_options)

                summary_writer.add_summary(summary, epoch)
                tf_score.append(tf_output)
            print(' Train_loss_score=', np.mean(tf_score))
            f.write(' Train_loss_score{:.4f}\n'.format(np.mean(tf_score)))
        summary_writer_test = tf.summary.FileWriter('./logs/test', session.graph)
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        print('TEST SET PERFORMANCE')
        f.write('TEST SET PERFORMANCE\n')
        y_test_pred, test_loss, summary = session.run([out_y_pred, loss, summary_op], feed_dict={in_X_tensors_batch: X_test,
                                                                            in_y_tensors_batch: y_test,
                                                                            is_training: False}, options=run_options)

        print('Test_loss_score=', test_loss)
        f.write('Test_loss_score={:4f}\n'.format(test_loss))

        with tf.name_scope('Accuracy'):
            acc = tf.equal(np.argmax(y_test_pred, axis=1).astype(np.int32), np.argmax(y_test, axis=1).astype(np.int32))

        summary_writer_test.add_summary(summary)


        y_test_pred_classified = np.argmax(y_test_pred, axis=1).astype(np.int32)
        y_test_true_classified = np.argmax(y_test, axis=1).astype(np.int32)
        print(classification_report(y_test_true_classified, y_test_pred_classified))
        f.write(classification_report(y_test_true_classified, y_test_pred_classified))
        print('{:2f}'.format(accuracy_score(y_test_true_classified, y_test_pred_classified)))
        f.write('Accuracy: {:2f}\n'.format(accuracy_score(y_test_true_classified, y_test_pred_classified)))
        cfm = confusion_matrix(y_test_true_classified, y_test_pred_classified)

        # plt.clf()
        # plt.imshow(cfm, interpolation='nearest', cmap=plt.cm.Blues)
        # plt.colorbar()
        # plt.tight_layout()
        # plt.savefig('cfm.png')
        # plt.show()

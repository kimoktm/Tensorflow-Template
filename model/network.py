# ============================================================== #
#                            Model                               #
#                                                                #
#                                                                #
# Neural Network tensorflow implementation                       #
# ============================================================== #


import tensorflow as tf


def build(features, labels, training = False):
    """
    Build network:
    ----------
    Args:
        features: Tensor, [batch_size, height, width, 3]
        labels: Tensor, [batch_size, num_classes]
        training: Boolean, in training mode or not (for dropout & bn)
    Returns:
        logits: Tensor, predicted classes [batch_size , num_classes]
    """

    # input_1 = tf.reshape(features["x"], [-1, 200, 200, 3])
    # conv1_1 = tf.layers.conv2d(inputs=input_1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    # conv1_2 = tf.layers.conv2d(inputs=conv1_1, filters=64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    # pool1   = tf.layers.max_pooling2d(inputs=conv1_2, pool_size=[2, 2], strides=2)

    # conv2_1 = tf.layers.conv2d(inputs=pool1  , filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    # conv2_2 = tf.layers.conv2d(inputs=conv2_1, filters=128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    # pool2   = tf.layers.max_pooling2d(inputs=conv2_2, pool_size=[2, 2], strides=2)

    # conv3_1 = tf.layers.conv2d(inputs=pool2  , filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    # conv3_2 = tf.layers.conv2d(inputs=conv2_1, filters=256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    # pool3   = tf.layers.max_pooling2d(inputs=conv3_2, pool_size=[2, 2], strides=2)
    # drop3   = tf.layers.dropout(inputs=pool3, rate=0.5, training= training)

    # # conv4_1 = tf.layers.conv2d(inputs=pool3  , filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    # # conv4_2 = tf.layers.conv2d(inputs=conv4_1, filters=512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu)
    # # pool4   = tf.layers.max_pooling2d(inputs=conv4_2, pool_size=[2, 2], strides=2)
    # # drop4   = tf.layers.dropout(inputs=pool4, rate=0.5, training= training)

    # flattend = tf.layers.flatten(inputs=drop3)
    # fully6   = tf.layers.dense(inputs=flattend, units=4096, activation=tf.nn.relu)
    # fully7   = tf.layers.dense(inputs=flattend, units=2048, activation=tf.nn.relu)
    # dropout7 = tf.layers.dropout(inputs=fully7, rate=0.5, training= training)
    # logits   = tf.layers.dense(inputs=dropout7, units=6)

    # output_size = labels.get_shape()[-1].value
    output_size = 5
    input_layer = tf.reshape(features["x"], [-1, 200, 200, 3])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    conv3 = tf.layers.conv2d(inputs=pool2, filters=128, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=[2, 2], strides=2)
    pool3_flat = tf.layers.flatten(pool3)

    dense1 = tf.layers.dense(inputs=pool3_flat, units=2048, activation=tf.nn.relu)
    dense = tf.layers.dense(inputs=dense1, units=1024, activation=tf.nn.relu)
    # dropout = tf.layers.dropout(inputs=dense, rate=0.5, training= training)

    logits = tf.layers.dense(inputs=dense, units=output_size)

    return logits


def loss(logits, labels):
    """
    Classification loss:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size,  num_classes]
        labels: Tensor, ground truth [batch_size, 1]

    Returns:
        loss: Classification loss
    """

    # loss = tf.losses.mean_squared_error(predictions=logits, labels=labels)

    rot_loss = tf.losses.mean_squared_error(predictions=logits[:3], labels=labels[:3])
    tra_loss = tf.losses.mean_squared_error(predictions=logits[3:5], labels=labels[3:5])
    loss = (3. / 10) * rot_loss + (2. / 15) * tra_loss

    tf.summary.scalar('rotation_loss', rot_loss)
    tf.summary.scalar('translation_loss', tra_loss)

    return loss


def train(loss, learning_rate):
    """
    Train opetation:
    ----------
    Args:
        loss: loss to use for training
        learning_rate: Float, learning rate

    Returns:
        train_op: Training operation
    """

    global_step = tf.train.get_global_step()
    decay_learning_rate = tf.train.exponential_decay(learning_rate, global_step,
                                           500, 0.9, staircase=True)
    tf.summary.scalar('learning_rate', decay_learning_rate)

    # optimizer = tf.train.AdamOptimizer(learning_rate=decay_learning_rate)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=decay_learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=decay_learning_rate, momentum=0.9)

    train_op = optimizer.minimize(
            loss=loss,
            global_step=global_step)

    return train_op


def predict(logits):
    """
    Prediction operation:
    ----------------
    Args:
        logits: Tensor, predicted    [batch_size,  num_classes]
    
    Returns:
        predicted_class: Tensor, predicted class   [batch_size, 1]
    """

    # multiplier = tf.constant(10**2, dtype=logits.dtype)
    # return tf.round(logits * multiplier) / multiplier

    return logits


def evaluate(logits, labels):
    """
    Classification accuracy:
    ----------
    Args:
        logits: Tensor, predicted    [batch_size,  num_classes]
        labels: Tensor, ground truth [batch_size, 1]

    Returns:
        accuracy: Classification accuracy
    """

    # eval_metric_ops = {
    #         "accuracy": tf.metrics.accuracy(
    #                 labels=labels, predictions=logits)}


    rot_loss = tf.metrics.mean_squared_error(predictions=logits[:3], labels=labels[:3])
    tra_loss = tf.metrics.mean_squared_error(predictions=logits[3:5], labels=labels[3:5])

    eval_metric_ops = {
            # "rmse": tf.metrics.root_mean_squared_error(
            #     labels=labels, predictions=logits),
            "rotation_loss": rot_loss,
            "translation_loss": tra_loss
            }

    return eval_metric_ops


def model_fn(features, labels, mode, params = None):
    """
    Model function:
    ----------
    Args:
        features: Tensor, [batch_size, height, width, 3]
        labels: Tensor, [batch_size, num_classes]
        mode: tf.mode, Train, predict or evaluate
        params: dict, optional parameters for the network

    Returns:
        estimator: tf.estimator, tf handler for training, prediction
    """

    logits = build(features, labels, mode == tf.estimator.ModeKeys.TRAIN)
    predictions = predict(logits)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    total_loss = loss(logits, labels)
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_op = train(total_loss, params['learning_rate'])
        return tf.estimator.EstimatorSpec(mode=mode, loss=total_loss, train_op=train_op)

    eval_metric_ops = evaluate(logits, labels)
    return tf.estimator.EstimatorSpec(
            mode=mode, loss=total_loss, eval_metric_ops=eval_metric_ops)

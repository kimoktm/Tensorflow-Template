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

    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
    conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
    
    conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[5, 5], padding="same", activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training= training)
    logits = tf.layers.dense(inputs=dropout, units=10)

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

    loss = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=labels)
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

    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())

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

    predictions = {
            # Generate predictions (for PREDICT and EVAL mode)
            "classes": tf.argmax(input=logits, axis=1),
            # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
            # `logging_hook`.
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    return predictions


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

    predictions = tf.argmax(input=logits, axis=1)
    eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                    labels=labels, predictions=predictions)}

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

import tensorflow as tf


def dqn_mask_loss(batch_data, y_pred):
    # The target is defined only for the action that was taken during the replay, hence the loss is computed based
    # only on this action's output
    batch_actions = tf.dtypes.cast(batch_data[:, 1], tf.int32)
    batch_true_q_values = batch_data[:, 0]
    mask = tf.one_hot(batch_actions, depth=y_pred.shape[1], dtype=tf.bool, on_value=True, off_value=False)
    batch_predicted_q_values = tf.boolean_mask(y_pred, mask)
    return tf.keras.losses.Huber()(batch_true_q_values, batch_predicted_q_values)
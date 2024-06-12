from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K
import tensorflow as tf

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

# Binary Crossentropy Dice Loss
def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# alphabcedice
def alphabcedice(y_true, y_pred, alpha=0.2):
    return alpha * binary_crossentropy(y_true, y_pred) + (1 - alpha) * dice_loss(y_true, y_pred)
    
#Focal
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.convert_to_tensor(y_true, dtype=tf.float32)
    y_pred = tf.convert_to_tensor(y_pred, dtype=tf.float32)

    # Focal Loss 계산
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true * alpha + (K.ones_like(y_true) - y_true) * (1 - alpha)
    p_t = y_true * y_pred + (K.ones_like(y_true) - y_true) * (1 - y_pred)
    fl = - alpha_t * K.pow((K.ones_like(y_true) - p_t), gamma) * K.log(p_t)
    
    return K.sum(fl, axis=-1)


def focal_dice_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    return focal_loss(y_true, y_pred, gamma, alpha) + dice_loss(y_true, y_pred)

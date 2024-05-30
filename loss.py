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

# ELL
def ell_loss(y_true, y_pred, alpha=0.2):
    return alpha * binary_crossentropy(y_true, y_pred) + (1 - alpha) * dice_loss(y_true, y_pred)


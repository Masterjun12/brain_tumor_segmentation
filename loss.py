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
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Focal Loss function.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param alpha: Alpha value for balancing class frequencies
    :param gamma: Gamma value for adjusting the rate of focus on hard examples
    :return: Focal Loss
    """
    # Compute Binary Crossentropy
    bce = binary_crossentropy(y_true, y_pred)
    
    # Compute Focal Loss
    p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
    alpha_factor = y_true * alpha + (1 - y_true) * (1 - alpha)
    modulating_factor = K.pow((1.0 - p_t), gamma)
    focal_loss = alpha_factor * modulating_factor * bce
    
    return focal_loss
#DiceFocal
def bce_dice_focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    """
    Combined Binary Crossentropy, Dice, and Focal Loss function.
    :param y_true: True labels
    :param y_pred: Predicted labels
    :param alpha: Alpha value for balancing class frequencies in Focal Loss
    :param gamma: Gamma value for adjusting the rate of focus on hard examples in Focal Loss
    :return: Combined loss
    """
    bce_dice = binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    focal = focal_loss(y_true, y_pred, alpha=alpha, gamma=gamma)
    return bce_dice + focal



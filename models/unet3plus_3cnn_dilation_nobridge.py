import tensorflow as tf
import tensorflow.keras.layers as L

def conv_block(x, num_filters, act=True):
    x = L.Conv2D(num_filters, kernel_size=3, padding="same")(x)

    if act == True:
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)

    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    x = conv_block(x, num_filters)
    x = conv_block(x, num_filters)

    p = L.MaxPool2D((2, 2))(x)
    return x, p

def f_model(input_shape, num_classes=4, deep_sup=False):
    """ Inputs """
    inputs = L.Input(input_shape, name="input_layer") ## (256 x 256 x 3)

    """ Encoder """
    e1, p1 = encoder_block(inputs, 64)
    e2, p2 = encoder_block(p1, 128)
    e3, p3 = encoder_block(p2, 256)
    e4, p4 = encoder_block(p3, 512)
    # print(e1.shape, e2.shape, e3.shape, e4.shape)

    """ Decoder 4 """
    e1_d4 = L.MaxPool2D((8, 8))(e1)
    e1_d4 = conv_block(e1_d4, 64)

    e2_d4 = L.MaxPool2D((4, 4))(e2)
    e2_d4 = conv_block(e2_d4, 64)

    e3_d4 = L.MaxPool2D((2, 2))(e3)
    e3_d4 = conv_block(e3_d4, 64)

    e4_d4 = conv_block(e4, 64)

    d4 = L.Concatenate()([e1_d4, e2_d4, e3_d4, e4_d4])
    d4 = conv_block(d4, 64*4)

    """ Decoder 3 """
    e1_d3 = L.MaxPool2D((4, 4))(e1)
    e1_d3 = conv_block(e1_d3, 64)

    e2_d3 = L.MaxPool2D((2, 2))(e2)
    e2_d3 = conv_block(e2_d3, 64)

    e3_d3 = conv_block(e3, 64)

    d4_d3 = L.UpSampling2D((2, 2), interpolation="bilinear")(d4)
    d4_d3 = conv_block(d4_d3, 64)

    d3 = L.Concatenate()([e1_d3, e2_d3, e3_d3, d4_d3])
    d3 = conv_block(d3, 64*4)

    """ Decoder 2 """
    e1_d2 = L.MaxPool2D((2, 2))(e1)
    e1_d2 = conv_block(e1_d2, 64)

    e2_d2 = conv_block(e2, 64)

    d3_d2 = L.UpSampling2D((2, 2), interpolation="bilinear")(d3)
    d3_d2 = conv_block(d3_d2, 64)

    d4_d2 = L.UpSampling2D((4, 4), interpolation="bilinear")(d4)
    d4_d2 = conv_block(d4_d2, 64)

    d2 = L.Concatenate()([e1_d2, e2_d2, d3_d2, d4_d2])
    d2 = conv_block(d2, 64*4)

    """ Decoder 1 """
    e1_d1 = conv_block(e1, 64)

    d2_d1 = L.UpSampling2D((2, 2), interpolation="bilinear")(d2)
    d2_d1 = conv_block(d2_d1, 64)

    d3_d1 = L.UpSampling2D((4, 4), interpolation="bilinear")(d3)
    d3_d1 = conv_block(d3_d1, 64)

    d4_d1 = L.UpSampling2D((8, 8), interpolation="bilinear")(d4)
    d4_d1 = conv_block(d4_d1, 64)

    d1 = L.Concatenate()([e1_d1, d2_d1, d3_d1, d4_d1])
    d1 = conv_block(d1, 64*4)

    """ Deep Supervision """
    if deep_sup == True:
        y1 = L.Conv2D(num_classes, kernel_size=1, padding="same")(d1)
        y1 = L.Activation("sigmoid")(y1)

        y2 = L.Conv2D(num_classes, kernel_size=1, padding="same")(d2)
        y2 = L.UpSampling2D((2, 2), interpolation="bilinear")(y2)
        y2 = L.Activation("sigmoid")(y2)

        y3 = L.Conv2D(num_classes, kernel_size=1, padding="same")(d3)
        y3 = L.UpSampling2D((4, 4), interpolation="bilinear")(y3)
        y3 = L.Activation("sigmoid")(y3)

        y4 = L.Conv2D(num_classes, kernel_size=1, padding="same")(d4)
        y4 = L.UpSampling2D((8, 8), interpolation="bilinear")(y4)
        y4 = L.Activation("sigmoid")(y4)

        outputs = [y1, y2, y3, y4]

    else:
        y1 = L.Conv2D(num_classes, kernel_size=1, padding="same")(d1)
        y1 = L.Activation("softmax")(y1)
        outputs = [y1]

    model = tf.keras.Model(inputs, outputs)
    return model
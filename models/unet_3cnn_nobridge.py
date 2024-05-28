from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.models import Model

def conv_block(x, num_filters):
    for _ in range(3): 
        x = Conv2D(num_filters, kernel_size=3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    return x

def encoder_block(x, num_filters):
    x = conv_block(x, num_filters)
    p = MaxPooling2D((2, 2))(x)
    return x, p

def decoder_block(x, skip_features, num_filters):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(x)
    x = concatenate([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def f_model(input_shape, num_classes=4):
    inputs = Input(input_shape)
    
    # Encoder
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)
    
    # Decoder
    d1 = decoder_block(p4, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)
    
    outputs = Conv2D(num_classes, (1, 1), padding="same", activation="softmax")(d4)
    model = Model(inputs, outputs, name="UNET")
    return model
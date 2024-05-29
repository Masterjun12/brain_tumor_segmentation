from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, PReLU
from tensorflow.keras.models import Model

def f_model(input_shape=(240, 240, 4), num_classes=4):
    inputs = Input(shape=input_shape)

    # Encoder
    conv1 = Conv2D(64, (3, 3), padding='same')(inputs)
    conv1 = PReLU()(conv1)  
    conv1 = Conv2D(64, (5, 5), padding='same')(conv1)
    conv1 = PReLU()(conv1)  
    conv1 = Conv2D(64, (1, 1), padding='same')(conv1)  # 1x1 conv
    conv1 = PReLU()(conv1)  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, (3, 3), padding='same')(pool1)
    conv2 = PReLU()(conv2)  
    conv2 = Conv2D(128, (5, 5), padding='same')(conv2)
    conv2 = PReLU()(conv2)  
    conv2 = Conv2D(128, (1, 1), padding='same')(conv2)  # 1x1 conv
    conv2 = PReLU()(conv2)  
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(256, (3, 3), padding='same')(pool2)
    conv3 = PReLU()(conv3)  
    conv3 = Conv2D(256, (5, 5), padding='same')(conv3)
    conv3 = PReLU()(conv3)  
    conv3 = Conv2D(256, (1, 1), padding='same')(conv3)  # 1x1 conv
    conv3 = PReLU()(conv3)  
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(512, (3, 3), padding='same')(pool3)
    conv4 = PReLU()(conv4)  
    conv4 = Conv2D(512, (5, 5), padding='same')(conv4)
    conv4 = PReLU()(conv4)  
    conv4 = Conv2D(512, (1, 1), padding='same')(conv4)  # 1x1 conv
    conv4 = PReLU()(conv4)  
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # Middle
    conv5 = Conv2D(1024, (3, 3), padding='same')(pool4) 
    conv5 = PReLU()(conv5)  
    conv5 = Conv2D(1024, (5, 5), padding='same')(conv5)
    conv5 = PReLU()(conv5)  
    conv5 = Conv2D(1024, (1, 1), padding='same')(conv5)  # 1x1 conv
    conv5 = PReLU()(conv5)  

    # Decoder
    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, (3, 3), padding='same')(up6)
    conv6 = PReLU()(conv6)  
    conv6 = Conv2D(512, (5, 5), padding='same')(conv6)
    conv6 = PReLU()(conv6)  
    conv6 = Conv2D(512, (1, 1), padding='same')(conv6)  # 1x1 conv
    conv6 = PReLU()(conv6)  

    up7 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, (3, 3), padding='same')(up7)
    conv7 = PReLU()(conv7)  
    conv7 = Conv2D(256, (5, 5), padding='same')(conv7)
    conv7 = PReLU()(conv7)  
    conv7 = Conv2D(256, (1, 1), padding='same')(conv7)  # 1x1 conv
    conv7 = PReLU()(conv7)  

    up8 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, (3, 3), padding='same')(up8)
    conv8 = PReLU()(conv8)  
    conv8 = Conv2D(128, (5, 5), padding='same')(conv8)
    conv8 = PReLU()(conv8)  
    conv8 = Conv2D(128, (1, 1), padding='same')(conv8)  # 1x1 conv
    conv8 = PReLU()(conv8)  

    up9 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, (3, 3), padding='same')(up9)
    conv9 = PReLU()(conv9)  
    conv9 = Conv2D(64, (5, 5), padding='same')(conv9)
    conv9 = PReLU()(conv9)  
    conv9 = Conv2D(64, (1, 1), padding='same')(conv9)  # 1x1 conv
    conv9 = PReLU()(conv9)  

    # Output
    outputs = Conv2D(num_classes, (1, 1), activation='softmax')(conv9)
    model = Model(inputs=inputs, outputs=outputs)
    return model

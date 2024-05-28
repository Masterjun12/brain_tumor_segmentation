from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose

def f_model(input_size=(240, 240, 4), num_classes=4):
    inputs = Input(input_size)
    
    # Contracting path
    conv1 = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2)(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', dilation_rate=2)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    
    conv2 = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=3)(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', dilation_rate=3)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation='relu', padding='same', dilation_rate=4)(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', dilation_rate=4)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation='relu', padding='same', dilation_rate=4)(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', dilation_rate=4)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', dilation_rate=8)(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', dilation_rate=8)(conv5)
    
    # Expansive path
    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=-1)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(up6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)
    
    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=-1)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(up7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)
    
    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=-1)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(up8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)
    
    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=-1)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(up9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
    
    conv10 = Conv2D(num_classes, 1, activation='softmax')(conv9)
    
    model = Model(inputs=[inputs], outputs=[conv10])
    return model

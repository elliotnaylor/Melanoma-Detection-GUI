
from keras.models import Model, Sequential
from keras.layers import Activation, Dense, GlobalAveragePooling2D, BatchNormalization, Dropout, Conv2D, Conv2DTranspose, AveragePooling2D, MaxPooling2D, UpSampling2D, Input, Reshape

class BorderModel:
    def __init__(self, img_size):
        self.img_size = img_size

    def getModelBorder(image_size):

        img_input = Input(shape=(image_size[0], image_size[1], image_size[2]))

        x = Conv2D(64, (3, 3), padding='same', name='conv1',strides= (1,1))(img_input)
        x = BatchNormalization(name='bn1')(x)
        x = Activation('relu')(x)
        x = Conv2D(64, (3, 3), padding='same', name='conv2')(x)
        x = BatchNormalization(name='bn2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        
        x = Conv2D(128, (3, 3), padding='same', name='conv3')(x)
        x = BatchNormalization(name='bn3')(x)
        x = Activation('relu')(x)
        x = Conv2D(128, (3, 3), padding='same', name='conv4')(x)
        x = BatchNormalization(name='bn4')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        
        x = Conv2D(256, (3, 3), padding='same', name='conv5')(x)
        x = BatchNormalization(name='bn5')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv6')(x)
        x = BatchNormalization(name='bn6')(x)
        x = Activation('relu')(x)
        x = Conv2D(256, (3, 3), padding='same', name='conv7')(x)
        x = BatchNormalization(name='bn7')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        
        x = Conv2D(512, (3, 3), padding='same', name='conv8')(x)
        x = BatchNormalization(name='bn8')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv9')(x)
        x = BatchNormalization(name='bn9')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv10')(x)
        x = BatchNormalization(name='bn10')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        
        x = Conv2D(512, (3, 3), padding='same', name='conv11')(x)
        x = BatchNormalization(name='bn11')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv12')(x)
        x = BatchNormalization(name='bn12')(x)
        x = Activation('relu')(x)
        x = Conv2D(512, (3, 3), padding='same', name='conv13')(x)
        x = BatchNormalization(name='bn13')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D()(x)
        
        x = Dense(1024, activation = 'relu', name='fc1')(x)
        x = Dense(1024, activation = 'relu', name='fc2')(x)


        pred = Reshape((192, 256))(x)

        model = Model(inputs=img_input, outputs=pred)

        return model
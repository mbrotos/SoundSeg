import tensorflow as tf
from keras.layers import Activation, Conv2D, BatchNormalization, Conv2DTranspose, Concatenate, MaxPooling2D, Input, Conv1D, Normalization

def get_model(img_size, num_classes=1):
    inputs = Input(shape=img_size + (1,))
    
    # TODO: set use_bias=False in Conv2D and Conv2DTranspose, because BatchNormalization is used
    conv1 = Conv2D(64, 3, strides=1, padding="same")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)

    conv2 = Conv2D(64, 3, strides=1, padding="same")(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation("relu")(conv2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, strides=1, padding="same")(pool1)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation("relu")(conv3)

    conv4 = Conv2D(128, 3, strides=1, padding="same")(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation("relu")(conv4)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(256, 3, strides=1, padding="same")(pool2)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation("relu")(conv5)

    conv6 = Conv2D(256, 3, strides=1, padding="same")(conv5)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation("relu")(conv6)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)

    conv7 = Conv2D(512, 3, strides=1, padding="same")(pool3)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation("relu")(conv7)

    conv8 = Conv2D(512, 3, strides=1, padding="same")(conv7)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation("relu")(conv8)

    pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)

    conv9 = Conv2D(1024, 3, strides=1, padding="same")(pool4)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation("relu")(conv9)

    conv10 = Conv2D(1024, 3, strides=1, padding="same")(conv9)
    conv10 = BatchNormalization()(conv10)
    conv10 = Activation("relu")(conv10)

    up1 = Conv2DTranspose(512, 2, strides=2, padding="same")(conv10)
    up1 = Concatenate()([up1, conv8])

    upconv1 = Conv2D(512, 3, strides=1, padding="same")(up1)
    upconv1 = BatchNormalization()(upconv1)
    upconv1 = Activation("relu")(upconv1)

    upconv2 = Conv2D(512, 3, strides=1, padding="same")(upconv1)
    upconv2 = BatchNormalization()(upconv2)
    upconv2 = Activation("relu")(upconv2)

    up2 = Conv2DTranspose(256, 2, strides=2, padding="same")(upconv2)
    up2 = Concatenate()([up2, conv6])

    upconv3 = Conv2D(256, 3, strides=1, padding="same")(up2)
    upconv3 = BatchNormalization()(upconv3)
    upconv3 = Activation("relu")(upconv3)

    upconv4 = Conv2D(256, 3, strides=1, padding="same")(upconv3)
    upconv4 = BatchNormalization()(upconv4)
    upconv4 = Activation("relu")(upconv4)

    up3 = Conv2DTranspose(128, 2, strides=2, padding="same")(upconv4)
    up3 = Concatenate()([up3, conv4])

    upconv5 = Conv2D(128, 3, strides=1, padding="same")(up3)
    upconv5 = BatchNormalization()(upconv5)
    upconv5 = Activation("relu")(upconv5)

    upconv6 = Conv2D(128, 3, strides=1, padding="same")(upconv5)
    upconv6 = BatchNormalization()(upconv6)
    upconv6 = Activation("relu")(upconv6)

    up4 = Conv2DTranspose(64, 2, strides=2, padding="same")(upconv6)
    up4 = Concatenate()([up4, conv2])

    upconv7 = Conv2D(64, 3, strides=1, padding="same")(up4)
    upconv7 = BatchNormalization()(upconv7)
    upconv7 = Activation("relu")(upconv7)

    upconv8 = Conv2D(64, 3, strides=1, padding="same")(upconv7)
    upconv8 = BatchNormalization()(upconv8)
    upconv8 = Activation("relu")(upconv8)

    output = Conv1D(num_classes, 1, activation="linear")(upconv8)

    # Define the model
    model = tf.keras.Model(inputs, output)
    return model
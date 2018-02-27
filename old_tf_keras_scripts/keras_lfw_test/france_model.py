from keras.layers import Input,merge
from keras.layers import Dense
from keras.layers import PReLU
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


def get_mymodel():

    inp = Input(shape=(112, 96, 3))

    # Block 1
    #x = Conv2D(32, (3, 3), padding='same', name='block1_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(inp)
    x = Conv2D(32, (3, 3), padding='same', name='block1_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(inp)
    x = BatchNormalization()(x)
    x =PReLU()(x)

    #x = Conv2D(64, (3, 3),  padding='same', name='block1_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    out1 = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    #x = Conv2D(64, (3, 3), padding='same', name='block2_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out1)
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out1)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(64, (3, 3), padding='same', name='block2_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(64, (3, 3), padding='same', name='block2_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out1], mode='sum')
    x = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(128, (3, 3), padding='same', name='block2_conv3',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv3', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    out2 = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    #x = Conv2D(128, (3, 3), padding='same', name='block3_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out2)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out2)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(128, (3, 3), padding='same', name='block3_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out2], mode='sum')
    out3 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(128, (3, 3), padding='same', name='block3_conv3',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out3)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv3', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out3)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(128, (3, 3), padding='same', name='block3_conv4',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(128, (3, 3), padding='same', name='block3_conv4', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out3], mode='sum')
    x = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block3_conv5',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv5', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    out4 = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out4)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out4)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out4], mode='sum')
    out5 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv3',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out5)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv3', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out5)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv4',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv4', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out5], mode='sum')
    out6 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv5',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out6)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv5', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out6)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv6',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv6', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out6], mode='sum')
    out7 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv7',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out7)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv7', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out7)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv8',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv8', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out7], mode='sum')
    out8 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv9',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out8)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv9', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out8)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(256, (3, 3), padding='same', name='block4_conv10',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(256, (3, 3), padding='same', name='block4_conv10', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out8], mode='sum')
    x = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(512, (3, 3), padding='same', name='block4_conv11',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv11', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    out9 = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    #Block 5
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv1',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out9)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out9)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv2',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out9], mode='sum')
    out10 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv3',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out10)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out10)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv4',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv4', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out10], mode='sum')
    out11 = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv5',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(out11)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv5', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(out11)
    x = BatchNormalization()(x)
    x = PReLU()(x)
    #x = Conv2D(512, (3, 3), padding='same', name='block5_conv6',kernel_initializer="he_normal",kernel_regularizer=l2(5e-4))(x)
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv6', kernel_initializer="glorot_uniform",kernel_regularizer=l2(5e-4))(x)
    x = BatchNormalization()(x)
    x = merge([x, out11], mode='sum')
    x = PReLU()(x)
    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    x = Flatten(name='flatten')(x)
    x = Dense(512, name='fc1')(x)
    x = BatchNormalization()(x)
    x = PReLU()(x)

    final_out=Dense(10575, activation='softmax', name='fc2')(x)

    model = Model(inp, final_out)

    return model
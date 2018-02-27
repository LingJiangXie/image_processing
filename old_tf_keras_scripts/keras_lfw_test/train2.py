import keras
import france_model

from keras.preprocessing.image import ImageDataGenerator

from keras.utils import multi_gpu_model
import lfw
import os
import numpy as np
import math
import facenet
import time
from keras.models import Model
import tensorflow as tf
'''
def normal(image):

    image = (image - 127.5) / 128.0

    return  image
'''

with tf.device('/cpu:0'):
    mymodel=france_model.get_mymodel()
    #mymodel.load_weights('epoch3.h5')

parallel_model = multi_gpu_model(mymodel, gpus=3)

parallel_model.compile(keras.optimizers.RMSprop(lr=0.1, rho=0.9, epsilon=1e-06), loss='categorical_crossentropy',metrics=['accuracy'])

data_dir = '/home/dany/Documents/CASIA-WebFace_aliged_116_100'


#可控的方式，就是在输入的图片已经是处理好的
batch_size = 360

datagen = ImageDataGenerator(
    
    width_shift_range=0.02,
    height_shift_range=0.018,
    horizontal_flip=True,
    #preprocessing_function=normal,
)

generator = datagen.flow_from_directory(
        data_dir,
        target_size=(112, 96),
        batch_size=batch_size,
        shuffle=True,
        #save_to_dir='/home/dany/Desktop/keras_image'
)


parallel_model.fit_generator(generator,epochs=5,steps_per_epoch=512)

mymodel.save('epoch5.h5')

lfw_pairs='/home/dany/Documents/workspace/facenet-master/data/pairs.txt'
lfw_dir='/home/dany/Documents/lfw_aliged_112_96'
lfw_file_ext='jpg'
lfw_nrof_folds=10
image_h=112
image_w=96
batch_size=128


pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs, lfw_file_ext)

embedding_size=512
nrof_images = len(paths)
nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
emb_array = np.zeros((nrof_images, embedding_size))

testmodel=france_model.get_mymodel()
testmodel.load_weights('epoch5.h5')

extrat_model=Model(inputs=testmodel.input,outputs=testmodel.get_layer('fc1').output)

for i in range(nrof_batches):
    start_index = i * batch_size
    end_index = min((i + 1) * batch_size, nrof_images)
    paths_batch = paths[start_index:end_index]
    images = facenet.load_data(paths_batch, image_h,image_w)

    t0 = time.time()
    y = extrat_model.predict_on_batch(images)
    emb_array[start_index:end_index, :] = y

    t1 = time.time()

    print('batch: ', i, ' time: ', t1 - t0)

from sklearn import metrics

tpr, fpr, accuracy, val, val_std, far = lfw.evaluate(emb_array,
                actual_issame, nrof_folds=lfw_nrof_folds)

print('Accuracy: %1.3f+-%1.3f' % (np.mean(accuracy), np.std(accuracy)))
print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' % (val, val_std, far))
auc = metrics.auc(fpr, tpr)
print('Area Under Curve (AUC): %1.3f' % auc)






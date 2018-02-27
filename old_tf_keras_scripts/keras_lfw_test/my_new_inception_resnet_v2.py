import tensorflow as tf
from keras.utils import multi_gpu_model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras import Model
from keras.callbacks import ModelCheckpoint,CSVLogger,TensorBoard

import lfw,os,math,time,facenet
import numpy as np
'''
def normal(image):

    image = (image - 127.5) / 128.0

    return  image
    


with tf.device('/cpu:0'):

    model = InceptionResNetV2(include_top=True, weights=None, input_shape=(224, 224, 3), pooling='avg', classes=10575)
    model.load_weights('epoch_6.h5')

    print(model.summary())

parallel_model = multi_gpu_model(model, gpus=3)

parallel_model.compile(RMSprop(lr=0.01, rho=0.9, epsilon=1e-06), loss='categorical_crossentropy',metrics=['accuracy'])

train_data_dir = '/home/dany/Documents/CASIA_train_232'
val_data_dir = '/home/dany/Documents/CASIA_val_232'

# 可控的方式，就是在输入的图片已经是处理好的
batch_size = 228

datagen = ImageDataGenerator(

    width_shift_range=0.018,
    height_shift_range=0.018,
    horizontal_flip=True,
    preprocessing_function=normal,
)

train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True,
    # save_to_dir='/home/dany/Desktop/keras_image'
)


val_generator = datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=batch_size,
    shuffle=True,
    )



#checkpoint = ModelCheckpoint("/home/dany/checkpoints/weights-improvement-{epoch:02d}-{val_loss:.4f}.hdf5", monitor='val_loss', verbose=1, save_best_only=False, mode='min')
logger=CSVLogger('/home/dany/checkpoints/log.csv', separator=',', append=False)
#tfboard=TensorBoard(log_dir='/home/dany/checkpoints/logs', histogram_freq=1, write_graph=True, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
callbacks_list = [logger]

parallel_model.fit_generator(train_generator,epochs=4,steps_per_epoch=845,validation_data=val_generator,validation_steps=62,callbacks=callbacks_list)

model.save('epoch_10.h5')
'''

lfw_pairs='/home/dany/Documents/workspace/facenet-master/data/pairs.txt'
lfw_dir='/home/dany/Documents/lfw_aliged_224_224'
lfw_file_ext='jpg'
lfw_nrof_folds=10
image_h=224
image_w=224
batch_size=128

pairs = lfw.read_pairs(os.path.expanduser(lfw_pairs))

paths, actual_issame = lfw.get_paths(os.path.expanduser(lfw_dir), pairs, lfw_file_ext)

embedding_size=1536
nrof_images = len(paths)
nrof_batches = int(math.ceil(1.0*nrof_images / batch_size))
emb_array = np.zeros((nrof_images, embedding_size))

testmodel=InceptionResNetV2(include_top=True, weights='imagenet',  input_shape=(224, 224, 3), pooling='avg')
#testmodel.load_weights('epoch_10.h5')

extrat_model=Model(inputs=testmodel.input,outputs=testmodel.get_layer('avg_pool').output)

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








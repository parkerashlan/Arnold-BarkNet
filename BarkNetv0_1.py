import _pickle as pickle
from tensorflow import keras

import DatasetMaker
import Image_Preprocessing

BATCH_SIZE = 50
DATA_PATH = './data/train_1_2'

dg = DatasetMaker.DatasetGenerator(path=DATA_PATH)

k_folds = dg.stratified_kfold_cv()

preprocess = Image_Preprocessing.BarkPreprocessing(DATA_PATH, k_folds)

folds = preprocess.make_train_validation_set()

train_folds = folds[0]
val_folds = folds[1]

img_shape = (224, 224, 3)  # shape after crop and channels is last

res_net = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=img_shape)
res_net.trainable = False

global_average_layer = keras.layers.GlobalAveragePooling2D()
output_layer = keras.layers.Dense(2, activation='softmax')

bark_resnet50 = keras.Sequential([
    res_net,
    global_average_layer,
    output_layer
])

bark_resnet50.compile(optimizer=keras.optimizers.Nadam(), loss=keras.losses.CategoricalCrossentropy(),
                      metrics=['categorical_accuracy'])

for i in range(1, k_folds[1]+1):

    dataset = preprocess.preprocessing(train_folds[i], val_folds[i], batch_size=BATCH_SIZE)

    train_fold_batch = dataset[0]
    val_fold_batch = dataset[1]

    callbacks = [keras.callbacks.ModelCheckpoint(f'./best_fold_{i}_model.h5', monitor='val_categorical_accuracy',
                                                 mode='max', save_best_only=True, verbose=1)]
    history = bark_resnet50.fit(train_fold_batch,
                                steps_per_epoch=len(train_folds[0])//BATCH_SIZE,
                                epochs=15,
                                validation_data=val_fold_batch,
                                validation_steps=len(val_folds[0])//BATCH_SIZE,
                                max_queue_size=300,
                                callbacks=callbacks)

    with open(f'./fold_{i}_results.pkl', 'wb') as file:
        file.write(pickle.dumps(history.history))




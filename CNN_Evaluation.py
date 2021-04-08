import tensorflow as tf
from tensorflow import keras

import Image_Preprocessing
import DatasetMaker

PATH = '../local_objects_PersonalBarkNet/data/test_data_1_2'
BATCH_SIZE = 32

dg = DatasetMaker.DatasetGenerator(PATH)

test_set = dg.stratified_kfold_cv(folds=1)


preprocess = Image_Preprocessing.BarkPreprocessing(PATH, test_set)

# need to make the testing set into a dataframe, will output as the validation set
# may try to fix to allow testing sets later
test_set_df = preprocess.make_train_validation_set()[1][0]

preprocessed_test_set = preprocess.preprocessing(test_df=test_set_df, batch_size=BATCH_SIZE, testing=True, flip=False)

BarkNet = keras.models.load_model('./early_stopping_models/best_fold_4_model.h5')

result = BarkNet.evaluate(preprocessed_test_set, steps=len(test_set_df)//BATCH_SIZE)


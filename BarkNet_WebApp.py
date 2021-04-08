import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import pandas as pd


@st.cache
def upload_bark(image_file):
    img = Image.open(image_file)
    return img


def random_crop(img, random_crop_size):
    """
    Randomly crops the image.

    :param
        img (NumPy Array): An image converted to a NumPy array with shape (width, height, channels).

        random_crop_size (int): Size to crop the image.

    :return:
       (NumPy Array) The cropped image as a NumPy array.
    """

    # Note: image_data_format is 'channel_last'
    assert img.shape[2] == 3

    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)

    return img[y:(y + dy), x:(x + dx), :]


@st.cache
def classify_bark(bark_image):

    bark_array = keras.preprocessing.image.img_to_array(bark_image, data_format='channels_last')

    bark_array_cropped = random_crop(bark_array, (224, 224))
    bark_array_cropped = np.expand_dims(bark_array_cropped, axis=0)
    bark_array_cropped = bark_array_cropped * (1./255.)

    class_num = {0: 'Ponderosa Pine', 1: 'California Incense Cedar',
                 2: 'WFIR', 3: 'VOAK'}

    model = keras.models.load_model('../PersonalBarkNet/models/best_fold_1_model.h5')
    probs = model(bark_array_cropped, training=False).numpy()

    prediction = np.argmax(probs)
    tree = class_num[int(prediction)]

    class_probs = dict()

    for i in range(len(probs[0])):
        class_probs[class_num[i]] = float(probs[0][i])

    # class_probs = [(class_num[i], float(probs[0][i])) for i in range(len(probs[0]))]

    return tree, class_probs


def learn_more(tree):

    if tree == 'California Incense Cedar':
        img = upload_bark('../local_objects_PersonalBarkNet/calincense.jpg')
        st.img(imag, width=400, height=400)


def main():

    IMG_PATH = './example_pics'
    image1 = upload_bark(IMG_PATH + '/2_0_CAL_OnePlus7_0022_04.jpg')
    image2 = upload_bark(IMG_PATH + '/1_0_PON_OnePlus7_0157_0009.jpg')

    st.sidebar.title('Tree Species Classification')

    choice = st.sidebar.selectbox('Upload Image (Pick an Image)', ['Choose an image', 'image1', 'image2'])

    st.sidebar.write('Choose an image of tree bark to learn the species of the tree and some interesting information. '
                     'After selecting the image then hit the button to classify the tree bark. After classifying the '
                     'tree then click on Learn More to learn about this species of tree.')

    button = st.sidebar.button('Classify Tree Bark')
    learn_more = st.sidebar.button('Learn More')

    if choice == 'image1' and not learn_more:
        tree_classification = classify_bark(image1)
        st.image(image1, width=400, height=400)

    if choice == 'image2' and not learn_more:
        tree_classification = classify_bark(image2)
        st.image(image2, width=400, height=400)

    if button:

        st.write('This tree is a', tree_classification[0], '.')
        st.subheader('Probabilities of tree species')
        st.dataframe(pd.DataFrame.from_dict(tree_classification[1], orient='index', columns=['Probabilites']))

    if learn_more:

        if choice == 'image1':
            tree_classification = classify_bark(image1)
            st.title('The '+tree_classification[0])
            img = upload_bark('../local_objects_PersonalBarkNet/calincense.jpg')
            st.image(img, width=400, height=400)

            st.write('Calocedrus decurrens, with the common names incense cedar and California incense-cedar '
                    '(syn. Libocedrus decurrens Torr.), is a species of conifer native to western North America. It '
                    'is the most widely known species in the genus, and is often simply called incense cedar without '
                    'the regional qualifier. Calocedrus decurrens is a large tree, typically reaching heights of 40–60 m '
                     '(130–195 ft) and a trunk diameter of up to 1.2 m (3.9 ft).')

        if choice == 'image2':
            tree_classification = classify_bark(image2)
            st.title('The ' + tree_classification[0])
            img = upload_bark('../local_objects_PersonalBarkNet/1200px-Pinus_ponderosa_15932.jpeg')
            st.image(img, width=400, height=400)

            st.write('Pinus ponderosa, commonly known as the ponderosa pine, bull pine, blackjack pine, '
                     'western yellow-pine, or filipinus pine is a very large pine tree species of variable habitat '
                     'native to mountainous regions of western North America. It is the most widely distributed pine '
                     'species in North America.')



if __name__ == "__main__":
    main()

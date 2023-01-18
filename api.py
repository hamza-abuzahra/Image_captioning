from flask import Flask
import numpy as np
import pickle
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras_preprocessing.sequence import pad_sequences


class CaptionGenerator:
    def __init__(self, model_name="8k"):
        self.CNNmodel = VGG16()
        self.CNNmodel = Model(inputs=self.CNNmodel.inputs, outputs=self.CNNmodel.layers[-2].output)
        self.model = keras.models.load_model(model_name)
        self.wrd_indx, self.indx_wrd, self.max_length = self.laod_data()

    def read_pickle(self, name):
        f = open(name, "rb")
        data = pickle.load(f)
        f.close()
        return data

    def load_data(self, model_name="8k"):
        if model_name == "8k":
            wrd_indx = self.read_pickle("word_indx8.pickle")
            indx_wrd = self.read_pickle("indx_word8.pickle")
            max_length = self.read_pickle("max_length8.pickle")
        return wrd_indx, indx_wrd, max_length
    
    def extract_features(self, image):
        ## deal with image format and stuff

        image = preprocess_input(image)
        features = self.CNNmodel.predict(image)
        return features  
        """
            filename = path + "/" + img_id
            img = image.load_img(filename, target_size=(224, 224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
        """

    def get_caption(self, image):
        feature = self.extract_features(image)
        in_text = "startsq"
        for i in range(self.max_length):
            in_seq = [self.wrd_indx[w] for w in in_text.split(" ")]
            in_seq = pad_sequences([in_seq], maxlen=self.max_length, padding="post")[0]
            in_seq = np.array([in_seq])
            pred = self.model.predict([feature, in_seq], verbose=0)
            pred = np.argmax(pred)
            word = self.indx_wrd.get(pred)
            if word is None:
                print("none")
                break
            in_text += " " + word
            if word == "endsq":
                break
        return in_text

# def main():

app = Flask(__name__)
cg = CaptionGenerator()

@app.route("/")
def hello():
    return "Hello World"

@app.route("/get_caption")
def return_caption():
    # get_caption()
    pass
    
# if __name__ == "__main__":
    # main()

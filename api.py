from flask import Flask, request
from flask_cors import CORS
import numpy as np
import pickle
import keras
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.models import Model
from keras_preprocessing import image 
from keras_preprocessing.sequence import pad_sequences
from PIL import Image

class CaptionGenerator:
    def __init__(self, model_name="8k"):
        self.CNNmodel = VGG16()
        self.CNNmodel = Model(inputs=self.CNNmodel.inputs, outputs=self.CNNmodel.layers[-2].output)
        self.model = keras.models.load_model(model_name)
        self.wrd_indx, self.indx_wrd, self.max_length = self.load_data()

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
    
    def extract_features(self, img):
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = self.CNNmodel.predict(img)
        return features  

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


app = Flask(__name__)
CORS(app)
cg = CaptionGenerator()
app.config['UPLOAD_FOLDER'] = 'uploads/'


@app.route("/")
def hello():
    return "Hello World"

@app.route("/get_caption", methods = ["POST"])
def return_caption():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    
    if file:
        img = Image.open(file   )
        img = img.convert("RGB")
        img = img.resize((224, 224))
        result = cg.get_caption(img)
    return result

if __name__ == "__main__":
    app.run(debug=True, port=1001)

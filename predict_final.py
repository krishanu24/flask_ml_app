import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime
from webbrowser import open_new_tab


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(224, 224))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()
    return img_tensor

def wrapStringInHTMLWindows(program, body):
    now = datetime.datetime.today().strftime("%Y%m%d-%H%M%S")
    txt = "<br/>"
    filename = program + '.html'
    f = open(filename,'w')
    wrapper = """<html>
    <head>
    <title>%s output - %s</title>
    </head>
    <body><p>Hello World Result: <a href=\"%s\">%s</a></p><p>%s</p>
    <img src="C:/Users/kragra/Documents/flask_ml_app/NORMAL-1017237-1.jpeg" width="200"
         height="80"></body>
    </html>"""
    whole = wrapper % (program, now, txt ,txt, body)
    f.write(whole)
    f.close()
    open_new_tab(filename)        
    
def display_in_html(pred):
    outstring = ""
    for s in pred:
      outstring += str(s)
      outstring += "<br />"
    wrapStringInHTMLWindows("PredictionResult", outstring)    
    
if __name__ == "__main__":
    # load model
    model = load_model("VGG16-basicModel.h5")
    # image path
    img_path ='C:/Users/kragra/Documents/flask_ml_app/Diag.png'
    #img_path = 'C:/Users/kragra/Documents/flask_ml_app/Diag1.png'    # dog
    #img_path = '/media/data/dogscats/test1/19.jpg'      # cat
    # load a single image
    new_image = load_image(img_path)
    # check prediction
    pred = model.predict(new_image)
    print(len(pred))
    display_in_html(pred)
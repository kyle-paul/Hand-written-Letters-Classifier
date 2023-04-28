import gradio as gr
import numpy as np

from keras.models import load_model
model = load_model('CNN_model.h5')

def recognize_digit(input):
  input = input.reshape((1,28,28,1))
  
  prediction = np.squeeze(model.predict(input))
  label = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z'] 
  output = dict(zip(label, prediction.tolist()))
  return output
  
gr.Interface(fn=recognize_digit, inputs=gr.Sketchpad(shape=(28, 28)), outputs="label", css=".input_interface {transform: scale(2);}", title="Hand-written Letters Classifier").launch(debug = True)
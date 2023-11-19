from keras.models import Model, load_model
import os
# os.environ['TF_KERAS'] = '1'
import keras2onnx

model = load_model('weight_mod_gray/model-057.h5')
onnx_model = keras2onnx.convert_keras(model,model.name)
keras2onnx.save_model(onnx_model,'lane_mod_gray.onnx')
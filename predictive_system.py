
import numpy as np

import pickle


loaded_model=pickle.load(open('trained_model.sav','rb'))

input_data=(2,1,2,1,21/6/2020,97,2,68,97,1,2,2,1,2,2,2,2,2,3,97,2,2,2,2)
input_data_as_numpy_array=np.asarray(input_data)
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshaped)
print(prediction)
if prediction[0]==0:
    print("The person is detected covid positive")
else:
    print("The person is detected covid negative")
    
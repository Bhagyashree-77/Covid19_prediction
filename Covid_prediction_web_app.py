import numpy as np
import pickle
import streamlit as st

loaded_model=pickle.load(open("C:\\Users\\BHAGYASHREE\\Covid_vaccine_prediction\\trained_model.sav",'rb'))

#creating a function for prediction
def Covid_prediction(input_data):
    input_data=(2,1,2,1,21/6/2020,97,2,68,97,1,2,2,1,2,2,2,2,2,3,97,2,2,2,2)
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshaped)
    print(prediction)
    if (prediction[0]==0):
        return 'The person is detected covid positive'
    else:
        return 'The person is detected covid negative'
    


def main():
    #Giving the title 
    st.title('Covid_Prediction_Web_App')

    #getting the input data
    PNEUMONIA,AGE,PREGNANT,DIABETES,COPD,ASTHMA,INMSUPR,HIPERTENSION,OTHER_DISEASE,CARDIOVASCULAR,OBESITY,RENAL_CHRONIC,TOBACCO,CLASIFFICATION_FINAL,ICU
    #USMER=st.text_input("Enter USER 1 or 2")
    #MEDICAL_UNIT=st.text_input(" 1 to 12 ")
    #SEX=st.text_input("Enter sex 1 for F& 2 for M")
    PATIENT_TYPE=st.text_input("Patient_Type 1 or 2")
    DATE_DIED=st.text_input(" DAte_Died DD/MM/YYYY")
    INTUBED=st.text_input("INTUBED 1,2,97")
    PNEUMONIA=st.text_input("PNEUMONIA 1,2")
    AGE=st.text_input("AGE")
    PREGNANT=st.text_input(" PREGNANT 97 OR 2")
    DIABETES=st.text_input("DIABETIES 1 OR 2")
    COPD=st.text_input("COPD 1 OR 2")
    ASTHMA=st.text_input("ASTHMA 1 OR 2")
    INMSUPR=st.text_input("INMSUPR 1 OR 2")
    HIPERTENSION=st.text_input("HIPERTENSION 1 OR 2")
    OTHER_DISEASE=st.text_input("OTHER_DISEASE 1 OR 2")
    CARDIOVASCULAR=st.text_input("CARDIOVASCULAR 1 OR 2")
    OBESITY=st.text_input("OBESITY 1 OR 2")
    RENAL_CHRONIC=st.text_input("RENAL_CHRONIC 1 OR 2")
    TOBACCO=st.text_input("TOBACCO 1 OR 2 ")
    CLASIFFICATION_FINAL=st.text_input(" CLASSIFICATION_FINAL =3,6,7")
    ICU=st.text_input(" ICU 97 OR 2")

    #CODE FOR PREDICTION 
    DIAGNOSIS=''
     
    #creating a button for prediction
    if st.button("Covid Test Result"):
        DIAGNOSIS=Covid_prediction([PNEUMONIA,AGE,PREGNANT,DIABETES,COPD,ASTHMA,INMSUPR,HIPERTENSION,OTHER_DISEASE,CARDIOVASCULAR,OBESITY,RENAL_CHRONIC,TOBACCO,CLASIFFICATION_FINAL,ICU])


    st.success(DIAGNOSIS)

if __name__=='__main__':
    main()



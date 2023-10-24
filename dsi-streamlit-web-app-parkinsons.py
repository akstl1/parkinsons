# Improt libraries
import streamlit as st
import pandas as pd
import joblib

# load our model pipeline object
model = joblib.load("model.joblib")

# insert a title for the app and instructions

st.title("Parkinsons Disease Prediction Model")
st.subheader("Enter patient information and submit for likelihood of parkinson's disease")


# chest pain input form

MDVP_Fo = st.number_input(
    label = "Enter the patients's recorded MDVP:Fo(Hz) [0-300]",
    min_value = 0.000000,
    max_value = 300.000000,
    value = 150.000000,
    step=0.01,
    format="%0.6f"
    )

MDVP_Fhi = st.number_input(
    label = "Enter the patients's recorded MDVP:Fhi(Hz) [50-700]",
    min_value = 50.0,
    max_value = 700.0,
    value = 500.0,
    step=0.01,
    format="%0.6f"
    )

MDVP_Flo = st.number_input(
    label = "Enter the patients's recorded MDVP:Flo(Hz) [50-300]",
    min_value = 50.0,
    max_value = 300.0,
    value = 100.0,
    step=0.01,
    format="%0.6f"
    )

MDVP_Jitter_pct = st.number_input(
    label = "Enter the patients's recorded MDVP:Jitter(%) [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.5,
    step=0.01,
    format="%0.6f"
    )

MDVP_Jitter_abs = st.number_input(
    label = "Enter the patients's recorded MDVP:Jitter(Abs) [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.0001,
    step=0.01,
    format="%0.6f"
    )

MDVP_RAP = st.number_input(
    label = "Enter the patients's recorded MDVP:RAP [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = .004,
    step=0.01,
    format="%0.6f"
    )

MDVP_PPQ = st.number_input(
    label = "Enter the patient's recorded MDVP:PPQ [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.001,
    step=0.01,
    format="%0.6f"
    )

Jitter_DDP = st.number_input(
    label = "Enter the patient's recorded Jitter:DDP [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.001,
    step=0.01,
    format="%0.6f"
    )

MDVP_Shimmer = st.number_input(
    label = "Enter the patient's recorded MDVP:Shimmer [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.05,
    step=0.01,
    format="%0.6f"
    )

MDVP_Shimmer_db = st.number_input(
    label = "Enter the patient's recorded MDVP:Shimmer(dB) [0-2]",
    min_value = 0.0,
    max_value = 2.0,
    value = 1.0,
    step=0.01,
    format="%0.6f"
    )

Shimmer_APQ3 = st.number_input(
    label = "Enter the patient's recorded Shimmer:APQ3 [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.007,
    step=0.01,
    format="%0.6f"
    )

Shimmer_APQ5 = st.number_input(
    label = "Enter the patient's recorded Shimmer:APQ5 [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.05,
    step=0.01,
    format="%0.6f"
    )

MDVP_APQ = st.number_input(
    label = "Enter the patient's recorded MDVP:APQ [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.1,
    step=0.01,
    format="%0.6f"
    )

Shimmer_DDA = st.number_input(
    label = "Enter the patient's recorded Shimmer:DDA [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.1,
    step=0.01,
    format="%0.6f"
    )

NHR = st.number_input(
    label = "Enter the patient's recorded NHR [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.2,
    step=0.01,
    format="%0.6f"
    )

HNR = st.number_input(
    label = "Enter the patient's recorded HNR [0-50]",
    min_value = 0.0,
    max_value = 50.0,
    value = 30.0,
    step=0.01,
    format="%0.6f"
    )

RPDE = st.number_input(
    label = "Enter the patient's recorded RPDE [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.5,
    step=0.01,
    format="%0.6f"
    )

DFA = st.number_input(
    label = "Enter the patient's recorded DFA [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.6,
    step=0.01,
    format="%0.6f"
    )

spread1 = st.number_input(
    label = "Enter the patient's recorded spread1 [-10 - 0]",
    min_value = -10.0,
    max_value = 0.0,
    value = -3.0,
    step=0.01,
    format="%0.6f"
    )

spread2 = st.number_input(
    label = "Enter the patient's recorded spread2 [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.3,
    step=0.01,
    format="%0.6f"
    )

D2 = st.number_input(
    label = "Enter the patient's recorded D2 [1-4]",
    min_value = 1.0,
    max_value = 4.0,
    value = 2.0,
    step=0.01,
    format="%0.6f"
    )

PPE = st.number_input(
    label = "Enter the patient's recorded PPE [0-1]",
    min_value = 0.0,
    max_value = 1.0,
    value = 0.3,
    step=0.01,
    format="%0.6f"
    )

#({"MDVP:Fo(Hz)" : [MDVP_Fo],
#"MDVP:Fhi(Hz)" : [MDVP_Fhi],
#"MDVP:Flo(Hz)" : [MDVP_Flo],
#"MDVP:Jitter(%)" : [MDVP_Jitter_pct],
#"MDVP:Jitter(Abs)" : [MDVP_Jitter_abs],
#"MDVP:RAP" : [MDVP_RAP],
#"MDVP:PPQ" : [MDVP_PPQ],
#"Jitter:DDP" : [Jitter_DDP],
#"MDVP:Shimmer" : [MDVP_Shimmer],
#"MDVP:Shimmer(dB)" : [MDVP_Shimmer_db],
#"Shimmer:APQ3" : [Shimmer_APQ3],
#"Shimmer:APQ5" : [Shimmer_APQ5],
#"MDVP:APQ" : [MDVP_APQ],
#"Shimmer:DDA" : [Shimmer_DDA],
#"NHR" : [NHR],
#"HNR" : [HNR],
#"RPDE" : [RPDE],
#"DFA" : [DFA],
#"spread1" : [spread1],
#"spread2" : [spread2],
#"D2" : [D2],
#"PPE" : [PPE]})

# submit inputs to model

if st.button("Submit For Prediction"):
    # store data into df for prediction
    new_data = pd.DataFrame({"MDVP:Fo(Hz)" : [MDVP_Fo],
    "MDVP:Fhi(Hz)" : [MDVP_Fhi],
    "MDVP:Flo(Hz)" : [MDVP_Flo],
    "MDVP:Jitter(%)" : [MDVP_Jitter_pct],
    "MDVP:Jitter(Abs)" : [MDVP_Jitter_abs],
    "MDVP:RAP" : [MDVP_RAP],
    "MDVP:PPQ" : [MDVP_PPQ],
    "Jitter:DDP" : [Jitter_DDP],
    "MDVP:Shimmer" : [MDVP_Shimmer],
    "MDVP:Shimmer(dB)" : [MDVP_Shimmer_db],
    "Shimmer:APQ3" : [Shimmer_APQ3],
    "Shimmer:APQ5" : [Shimmer_APQ5],
    "MDVP:APQ" : [MDVP_APQ],
    "Shimmer:DDA" : [Shimmer_DDA],
    "NHR" : [NHR],
    "HNR" : [HNR],
    "RPDE" : [RPDE],
    "DFA" : [DFA],
    "spread1" : [spread1],
    "spread2" : [spread2],
    "D2" : [D2],
    "PPE" : [PPE]})
    
    # apply model pipeline to the input data and extract probability prediction
    pred_proba = model.predict_proba(new_data)[0][1]
    
    # output prediction
    st.subheader(f"Based on these customer attributes, our model predicts a parkinson's disease probability of {pred_proba: .0%}")
import gradio as gr 
import skops.io as sio
import pandas as pd

pipe = sio.load("./Model/personality_pipeline.skops",trusted=["numpy.dtype","sklearn.compose._column_transformer._RemainderColsList"])
print("Succesfully loaded the model")

def predict_personality(Time_spent_Alone,Stage_fear,Social_event_attendance,Going_outside,Drained_after_socializing,Friends_circle_size,Post_frequency):
    """Predict personality based on social features.

    Args:
        + Time_spent_Alone: Hours spent alone daily (0-11).
        + Stage_fear: Presence of stage fright (Yes/No).
        + Social_event_attendance: Frequency of social events (0-10).
        + Going_outside: Frequency of going outside (0-7).
        + Drained_after_socializing: Feeling drained after socializing (Yes/No).
        + Friends_circle_size: Number of close friends (0-15).
        + Post_frequency: Social media post frequency (0-10).

    Returns:
        str: Predicted drug label
    """
    features = [Time_spent_Alone,Stage_fear,Social_event_attendance,Going_outside,Drained_after_socializing,Friends_circle_size,Post_frequency]
    columns = [
        "Time_spent_Alone",
        "Stage_fear",
        "Social_event_attendance",
        "Going_outside",
        "Drained_after_socializing",
        "Friends_circle_size",
        "Post_frequency"
        ]
    
    df_input = pd.DataFrame([features], columns=columns)
    predicted_personality = pipe.predict(df_input)[0]
    return f"Predicted personality: {str(predicted_personality).upper()}"

inputs=[
    gr.Slider(0,11,step=1,label="Time_spent_Alone",info="How many hours do you spend alone every day? (0-11)"),
    gr.Radio(["Yes","No"],label="Stage_fear",info="Are you afraid of standing in front of crowds? (Yes/No)."),
    gr.Slider(0,10,step=1,label="Social_event_attendance",info="How frequently do you participate in social events? (0-10)"),
    gr.Slider(0,7,step=1,label="Going_outside",info="How many days per week do you go outside? (0-7)"),
    gr.Radio(["Yes","No"],label="Drained_after_socializing",info="Do you feel exhausted after parties or interacting with many people? (Yes/No)."),
    gr.Slider(0,15,step=1,label="Friends_circle_size",info="How many close friends do you have? (0-15)"),
    gr.Slider(0,10,step=1,label="Post_frequency",info="How often do you post on social media? (0-10)"),
]
outputs=[gr.Label(num_top_classes=5)]

examples=[
    [4.0,"No",4.0,6.0,"No",13.0,5.0],
    [9.0,"Yes",0.0,0.0,"Yes",0.0,3.0],
    [9.0,"Yes",1.0,2.0,"Yes",5.0,2.0]
]

title="Personality Classification"
description="Enter your social features to determine whether you are an Introvert or Extrovert"
article="This app is for my MLOps with CI/CD pipe line, pretty cool right?"

gr.Interface(
    fn=predict_personality,
    inputs=inputs,
    outputs=outputs,
    examples=examples,
    title=title,
    description=description,
    article=article,
    theme=gr.themes.Soft()
).launch()
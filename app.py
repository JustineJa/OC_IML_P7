from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, Flatten, Dense, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Sequential
import streamlit as st
from tensorflow.keras.applications import ConvNeXtTiny, ResNet50 
from tensorflow.keras.applications.convnext import preprocess_input, decode_predictions
import plotly.express as px
import pandas as pd

# Afficher la bannière
banniere_path = 'banniere.png'
image_ban = Image.open(banniere_path)
st.image(image_ban, caption='Bienvenue', use_column_width=True)

# Interface Streamlit
st.title("Classification d'images")
st.header("Contexte")
st.write("L'univers du Machine Learning évolue rapidement.")
st.write("Nous allons ici réaliser une veille technologique pour identifier un algorithme qui pourrait s'avérer plus performant que le réseau de neurone Resnet que nous avons étudié au projet 6.")

st.header("Nos ressources :")
st.write("Pour nos travaux nous nous basons sur le dataset fourni par l'université de Standford.")
st.write("Vous trouverez ci dessous la répartition des images par race dans ce dataset :")
df_races = pd.read_csv('races.csv')
# Créer un graphique interactif avec Plotly
fig_races = px.bar(df_races, x="index", y="num_pictures", title="Répartition des photos par race")
# Afficher le graphique dans Streamlit
st.plotly_chart(fig_races)

st.header("Nos résultats :")
st.write("Dans le cadre de notre étude nous avons souhaité tester la performance de l'algorithme ConvNeXtTiny par rapport à ResNet50.")
st.write("Nous avons réalisé nos tests pour différentes cardinalités (entre 5 et 40) de classes :")
df_results = pd.read_csv('results_vf.csv')
# Créer un graphique interactif avec Plotly
fig_results = px.line(df_results, x="Nb_races", y="Accuracy",color = "Model", title="Accuracy par algorithme")
# Afficher le graphique dans Streamlit
st.plotly_chart(fig_results)

# Créer un graphique interactif avec Plotly
fig_time = px.line(df_results, x="Nb_races", y="Temps (sec)",color = "Model", title="Temps par algorithme")
# Afficher le graphique dans Streamlit
st.plotly_chart(fig_time)



st.header("Testons notre algorithme !")
st.write("Téléchargez une image à classifier")

# Télécharger une image via Streamlit
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])
list_races = ['Lévrier Afghan', 'Bouledogue français', 'Bichon maltais', 'Loulou de Poméranie', 'Deerhound']

# Charger le modèle
@st.cache_resource
def load_model(model_path):
    model = keras.models.load_model(model_path)
    return model

# Fonction de prédiction
def predict(model, img):
    # Redimensionner l'image à la taille attendue (224, 224 pour ConvNextTiny)
    img = img.resize((224, 224))

    # Convertir l'image en tableau numpy
    img_array = np.array(img)

    # Ajouter une dimension supplémentaire pour correspondre à l'entrée du modèle
    img_array = np.expand_dims(img_array, axis=0)

    # Prétraiter l'image pour le modèle ConvNextTiny
    img_array = preprocess_input(img_array)

    # Faire la prédiction
    prediction = model.predict(img_array)
    # Convert predictions to class labels
    predicted_class = int(np.argmax(prediction, axis=-1))
    
    # Invert the predicted class indices to actual labels
    labeled_prediction = list_races[predicted_class]
    
    return labeled_prediction

# Charger le modèle
model = load_model('model.h5')

if uploaded_file is not None:
    # Afficher l'image téléchargée
    image = Image.open(uploaded_file)
    st.image(image, caption='Image téléchargée', use_column_width=True)

    # Prédire la classe de l'image
    st.write("Classification en cours...")
    labeled_prediction = predict(model, image)
    
    # Afficher les résultats
    st.write("Voici le résultat de la classification :")
    st.write(f"La race est: {labeled_prediction}")

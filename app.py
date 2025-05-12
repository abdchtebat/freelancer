import pickle
import streamlit as st
import base64
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Chemin de ton image téléchargée
image_path = "/Users/abderhmanchtebat/Documents/2.png"  # Mets le chemin correct ici

# Convertir l'image en base64
with open(image_path, "rb") as image_file:
    img = image_file.read()
    img_b64 = base64.b64encode(img).decode()

# Ajouter l'image en arrière-plan
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] {{
background-image: url("data:image/jpeg;base64,{img_b64}");
background-size: cover;
background-position: center;
background-repeat: no-repeat;
}}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)


# Charger le modèle
with open("/Users/abderhmanchtebat/Desktop/AI M107/streamlit/flask/random_forest_model.pkl", "rb") as file:
    model = joblib.load(file)

# Initialiser LabelEncoder
label_encoder = LabelEncoder()

# Charger le fichier CSV
ht = pd.read_csv("/Users/abderhmanchtebat/Desktop/AI M107/streamlit/flask/freelancer_earnings_bd.csv")

# Titre de l'application
st.markdown("<h1 style='color: #0f0503;'>Freelancer Earnings Prediction</h1>", unsafe_allow_html=True)

# Bouton pour afficher les champs de formulaire
if st.button('Make Model Prediction'):
    st.session_state.form_shown = True

# Créer un conteneur pour le formulaire
if 'form_shown' in st.session_state and st.session_state.form_shown:
    with st.container():

        st.markdown("<h2 style='color: #0f0503;'>Veuillez entrer les informations du freelancer</h2>", unsafe_allow_html=True)

        # Inputs pour les informations du freelancer
        job_category = st.selectbox('Catégorie de travail', ht['Job_Category'].unique())
        platform = st.selectbox('Plateforme', ht['Platform'].unique())
        experience_level = st.selectbox('Niveau d\'expérience', ht['Experience_Level'].unique())
        client_region = st.selectbox('Région du client', ht['Client_Region'].unique())
        payment_method = st.selectbox('Méthode de paiement', ht['Payment_Method'].unique())
        job_completed = st.number_input('Nombre de jobs terminés', min_value=0)
        earnings_usd = st.number_input('Gains en USD', min_value=0.0, step=0.01)
        hourly_rate = st.number_input('Taux horaire', min_value=0.0, step=0.01)
        job_success_rate = st.number_input('Taux de réussite du job (%)', min_value=0.0, max_value=100.0)
        client_rating = st.number_input('Note du client', min_value=0.0, max_value=5.0)
        job_duration_days = st.number_input('Durée du job (en jours)', min_value=0)
        project_type = st.selectbox('Type de projet', ht['Project_Type'].unique())
        rehire_rate = st.number_input('Taux de réembauche (%)', min_value=0.0, max_value=100.0)
        marketing_spend = st.number_input('Dépenses marketing', min_value=0.0, step=0.01)

        # Fonction pour encoder les valeurs catégorielles
        def encode_feature(value, possible_values):
            if value in possible_values:
                return label_encoder.fit_transform([value])[0]
            else:
                st.error(f"Valeur '{value}' non reconnue dans les catégories possibles.")
                return 0  # Valeur par défaut si non trouvée

        # Bouton "Predict" pour exécuter la prédiction
        predict_button = st.button('Predict')

        if predict_button:
            # Encodage des variables catégorielles
            job_category_encoded = encode_feature(job_category, ht['Job_Category'].unique())
            platform_encoded = encode_feature(platform, ht['Platform'].unique())
            experience_level_encoded = encode_feature(experience_level, ht['Experience_Level'].unique())
            client_region_encoded = encode_feature(client_region, ht['Client_Region'].unique())
            payment_method_encoded = encode_feature(payment_method, ht['Payment_Method'].unique())
            project_type_encoded = encode_feature(project_type, ht['Project_Type'].unique())

            # Préparation des données d'entrée
            features = np.array([[
                job_category_encoded, platform_encoded, experience_level_encoded,
                client_region_encoded, payment_method_encoded, job_completed,
                earnings_usd, hourly_rate, job_success_rate, client_rating,
                job_duration_days, project_type_encoded, rehire_rate, marketing_spend
            ]])

            # Vérification de la correspondance des dimensions
            if features.shape[1] != model.n_features_in_:
                st.error(f"Le nombre de caractéristiques d'entrée est incorrect. Le modèle attend {model.n_features_in_} caractéristiques.")
            else:
                # Prédiction
                prediction = model.predict(features)

                # Affichage du résultat
                if prediction[0] == 1:  # High Rehire Rate (likely higher earnings)
                    st.success(f"Le freelancer devrait probablement générer des gains Faibles.")
                else:  # Low Rehire Rate (likely lower earnings)
                    st.success(f"Le freelancer devrait probablement générer des gains élevés.")


# Section de visualisation
if st.button('Visualisation'):
    # Graphique 1 : Distribution des gains en USD des freelances (avec densité)
    plt.figure(figsize=(10, 6))
    sns.histplot(ht['Earnings_USD'], bins=30, kde=True, color='skyblue')
    plt.title('Distribution des gains en USD des freelances (avec densité)')
    plt.xlabel('Gains (USD)')
    plt.ylabel('Fréquence')
    plt.grid(True)
    st.pyplot(plt)

    # Graphique 2 : Répartition des freelances par niveau d'expérience
    plt.figure(figsize=(8, 6))
    experience_counts = ht['Experience_Level'].value_counts()
    plt.pie(experience_counts, labels=experience_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue', 'lightgreen'])
    plt.title('Répartition des freelances par niveau d\'expérience')
    st.pyplot(plt)

    # Graphique 3 : Nombre de freelances par catégorie de travail
    plt.figure(figsize=(12, 6))
    sns.countplot(data=ht, x='Job_Category', palette='Set2', order=ht['Job_Category'].value_counts().index)
    plt.title('Nombre de freelances par catégorie de travail')
    plt.xlabel('Catégorie de travail')
    plt.ylabel('Nombre de freelances')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Graphique 4 : Heatmap des gains moyens par catégorie de travail et niveau d'expérience
    pivot_table = pd.pivot_table(ht, values='Earnings_USD', index='Job_Category', columns='Experience_Level', aggfunc='mean')
    sns.heatmap(pivot_table, annot=True, cmap='YlGnBu', fmt='.2f', linewidths=0.5)
    plt.title('Heatmap des gains moyens par catégorie de travail et niveau d\'expérience')
    st.pyplot(plt)

    # Graphique 5 : Distribution des taux de succès des missions
    plt.figure(figsize=(10, 6))
    sns.histplot(ht['Job_Success_Rate'], bins=30, kde=True, color='lightgreen')
    plt.title('Distribution des taux de succès des missions')
    plt.xlabel('Taux de succès (%)')
    plt.ylabel('Fréquence')
    plt.grid(True)
    st.pyplot(plt)

    # Graphique 6 : Répartition des freelances par région du client
    plt.figure(figsize=(8, 6))
    client_region_counts = ht['Client_Region'].value_counts()
    plt.pie(client_region_counts, labels=client_region_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightcoral', 'lightblue', 'lightgreen', 'lightyellow'])
    plt.title('Répartition des freelances par région du client')
    st.pyplot(plt)

    # Graphique 7 : Nombre de freelances par méthode de paiement
    plt.figure(figsize=(10, 6))
    sns.countplot(data=ht, x='Payment_Method', palette='Set3', order=ht['Payment_Method'].value_counts().index)
    plt.title('Nombre de freelances par méthode de paiement')
    plt.xlabel('Méthode de paiement')
    plt.ylabel('Nombre de freelances')
    plt.xticks(rotation=45)
    st.pyplot(plt)

    # Graphique 8 : Répartition des freelances par type de projet
    plt.figure(figsize=(8, 6))
    project_type_counts = ht['Project_Type'].value_counts()
    plt.pie(project_type_counts, labels=project_type_counts.index, autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightgreen'])
    plt.title('Répartition des freelances par type de projet')
    st.pyplot(plt)

    # Graphique 9 : Distribution de la durée des projets (en jours)
    plt.figure(figsize=(10, 6))
    sns.histplot(ht['Job_Duration_Days'], bins=30, kde=True, color='lightyellow')
    plt.title('Distribution de la durée des projets (en jours)')
    plt.xlabel('Durée des projets (jours)')
    plt.ylabel('Fréquence')
    plt.grid(True)
    st.pyplot(plt)







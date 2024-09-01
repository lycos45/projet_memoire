import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from time import time
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline, make_union
import statsmodels.formula.api as smf
import statsmodels.api as sm
from PIL import Image
import streamlit as st
from io import BytesIO
import base64
@st.cache_data
def wrangle(filepath):
    #read csv
    df= pd.read_csv(filepath,sep=';', na_values=["", " ", "  ", "   ", "    ", np.nan], low_memory=False)

    # remplacer le nom d'un variable
    df['pratique_cultures_fourrageres']= df['1. pratique_cultures_fourrageres']
    df.drop(columns= "1. pratique_cultures_fourrageres", inplace = True)

    #creation du variables rendement et supression des autre rendement
    def merge_columns(row):
        cols = [row['rend_moyen_arachide'], row['rend_moyen_mil'],row['rend_moyen_niebe'],row['rend_moyen_mais'],row['rend_moyen_sorgho'],row['rend_moyen_fonio'],row['rend_moyen_riz_irrigue'],row['rend_moyen_riz_pluvial']]
        non_nan_values = [v for v in cols if pd.notna(v)]

        if len(non_nan_values) == 0:
            return np.nan
        elif len(non_nan_values) >= 2:
            return 0
        else:
            return non_nan_values[0]
    # Appliquer la fonction à chaque ligne du DataFrame
    df['rendement'] = df.apply(merge_columns, axis=1)
    #supression des collonne rendement
    df.drop(columns=['rend_moyen_arachide', 'rend_moyen_mil', 'rend_moyen_niebe', 'rend_moyen_mais', 'rend_moyen_sorgho', 'rend_moyen_fonio', 'rend_moyen_riz_irrigue', 'rend_moyen_riz_pluvial'], inplace=True)
    catcol_convert =['id_reg','culture_principale','irrigation','pastèque_fin_saison','rotation_cultures','culture_principale_précédente_2020_2021','méthode_de_culture','culture_secondaire','varietes_arachides_CP','origine_semence_CP',
    'variete_riz_CP','type_semence_CP','mois_semis_CP','semaine_semis_CP','etat_produit_CP','mois_previsionnel_recolte_CP','varietes_arachides_CS','origine_semence_CS','variete_riz_CS','type_semence_CS','mois_semis_CS',
    'semaine_semis_CS','etat_produit_CS','mois_previsionnel_recolte_CS','utilisation_matiere_org_avant_semis', 'matieres_organiques', 'utilisation_matiere_miner_avant_semis','matieres_minerales','prevision_epandage_engrais_min_avant_recolte',
    'type_engrais_mineral_epandage_avant_recolte','utilisation_produits_phytosanitaires','utilisation_residus_alimentation_animaux','type_residus_alimentation_animaux','type_labour', 'type_couverture_sol_interculture',
    'type_installation','pratiques_conservation_sol','contraintes_production','type_materiel_preparation_sol','type_materiel_semis','type_materiel_entretien_sol','type_materiel_récolte',
    'type_culture','rendement_arachide','rendement_mil','rendement_niebe','rendement_mais','rendement_riz_irrigue','rendement_sorgho','rendement_fonio','rendement_riz_pluvial','pratique_cultures_fourrageres',
    'cultures_fourrageres_exploitees','production_forestiere_exploitation']

    numcol_convert=['rendement','superficie_parcelle','pourcentage_superficie_pastèque','pourcentage_CP','quantite_semence_CP','quantite_semence_sub_CP','quantite_semence_marche_specialisee_CP ','quantite_semence_reserve_personnelle_CP','quantite_semence_don_CP',
    'quantite_CP','pourcentage_CS','quantite_semence_CS','quantite_semence_sub_CS','quantite_semence_marche_specialisee_CS','quantite_semence_reserve_personnelle_CS','quantite_semence_don_CS',
    'quantite_CS','quantite_NPK_epandage_avant_recolte','quantite_urée_epandage_avant_recolte','nombre_pieds_arachide_compte','nombre_pieds_arachide_recoltes','poids_total_gousses_arachide_recoltés','poids_moyen_arachide_en_gramme',
    'poids_recolte_arachide','nombre_pieds_mil_compte','nombre_epis_potentiel_maturite_mil','nombre_epis_prélevé_mil','poids_total_graines_battage_sechage_mil','poids_moyen_mil_en_gramme','poids_recolte_mil',
    'nombre_pieds_niebe_compte','nombre_gousses_niebe_3_pieds','nombre_gousses_niebe_par_pieds','nombre_total_gousses_niebe','nombre_gousses_niebe_preleve','poids_total_gousses_niebe_apres_egrainage','poids_moyen_gousses_niebe',
    'poids_total_niebe_de_la_recolte','nombre_pieds_mais_compte','nombre_epis_potentiel_maturite_mais','nombre_epis_mais_preleve','poids_total_graines_battage_sechage_mais',
    'poids_moyen_mais_en_gramme','poids_recolte_mais','nombre_sorgho_compte','nombre_epis_potentiel_maturite_sorgho','nombre_epis_sorgho_preleve','poids_total_graines_battage_sechage_sorgho','poids_moyen_sorgho_en_gramme',
    'poids_recolte_sorgho','poids_recolte_fonio','nombre_pieds_riz_irrigue_compte','nombre_epis_potentiel_maturite_riz_irrigue',
    'nombre_epis_riz_irrigue_preleve','poids_total_graines_battage_sechage_riz_irrigue','poids_moyen_riz_irrigue_en_gramme','poids_recolte_riz_irrigue','nombre_pieds_riz_pluvial_compte','nombre_epis_potentiel_maturite_riz_pluvial',
    'nombre_epis_riz_pluvial_preleve','poids_total_graines_battage_sechage_riz_pluvial','poids_moyen_riz_pluvial_en_gramme','poids_recolte_riz_pluvial','superficie(ha)_cultures_fourrageres']

    #conversion des types
    #colonne numerique
    for col in numcol_convert:
        # Remplacer les virgules par des points if the column is of type object (string)
        if df[col].dtype == 'object': # Check if column is of type object
            df[col] = df[col].str.replace(',', '.')
        df[col] = df[col].astype('float64')
    #pour les colonne category
    for col in catcol_convert:
        df[col] = df[col].astype('category')


    #suprimer les variable qui on plus de 50% des donnees manquante
    missing_percent = df.drop(columns="rendement").isnull().mean() * 100
    columns_to_drop = missing_percent[missing_percent > 50].index.tolist()
    df.drop(columns=columns_to_drop, inplace =True)



    # Fonction pour imputer les valeurs manquantes avec HistGradientBoostingRegressor
    def impute_missing_values(data, column, model):
        # Séparer les lignes avec des valeurs manquantes et sans valeurs manquantes
        data_missing = df[df[column].isnull()]
        data_not_missing = df[df[column].notnull()]

        # Séparer les caractéristiques (X) et la cible (y)
        X_missing = data_missing.drop(columns=[column])
        X_not_missing = data_not_missing.drop(columns=[column])
        y_not_missing = data_not_missing[column]

        # Ajuster le modèle sur les données non manquantes
        model.fit(X_not_missing, y_not_missing)

        # Prédire les valeurs manquantes
        y_missing_pred = model.predict(X_missing)

        # Remplacer les valeurs manquantes par les valeurs prédites
        df.loc[df[column].isnull(), column] = y_missing_pred
        return df

    # Choisir un modèle de régression
    model = HistGradientBoostingRegressor()

    #   Get numeric columns AFTER filtering the DataFrame
    numeric_columns = df.select_dtypes(include=['float64']).columns.tolist()

    # Imputer les valeurs manquantes pour chaque colonne numérique
    for column in numeric_columns: # Use the updated list of numeric columns
        df = impute_missing_values(df, column, model)


    #pour les variable categoriel
    # Function to impute missing categorical values with RandomForestClassifier
    def impute_categorical_values(data, column):
        # Separate rows with and without missing values
        data_missing = df[df[column].isnull()]
        data_not_missing = df[df[column].notnull()]

        # Separate features (X) and target (y)
        X_missing = data_missing.drop(columns=[column])
        X_not_missing = data_not_missing.drop(columns=[column])
        y_not_missing = data_not_missing[column]

        # Impute missing values in X_missing and X_not_missing using SimpleImputer
        imputer = SimpleImputer(strategy='most_frequent')  # Use most frequent strategy for categorical features
        X_missing_imputed = imputer.fit_transform(X_missing)
        X_not_missing_imputed = imputer.transform(X_not_missing)

        # Choose a classification model
        model = RandomForestClassifier()

        # Fit the model on the non-missing data
        model.fit(X_not_missing_imputed, y_not_missing)  # Use imputed data

        # Predict the missing values
        y_missing_pred = model.predict(X_missing_imputed)  # Use imputed data

        # Replace the missing values with the predicted values
        data.loc[data[column].isnull(), column] = y_missing_pred

        return data
    df.drop_duplicates(inplace= True)
    #origine_semence_CP  type_semence_CP pratiques_conservation_sol
    df.drop(columns=['origine_semence_CP','type_semence_CP','pratiques_conservation_sol' ], inplace=True, axis=1)
    # Impute missing values for each categorical column
    for column in (df.select_dtypes(include=['category']).columns.tolist()):
        df = impute_categorical_values(df, column)


    #supression des variables qui n'ont pas d'influence sur le model
    df.drop(columns= ['pastèque_fin_saison', 'contraintes_production', 'type_materiel_entretien_sol'])

    #gestion des outlier
    #definition de fonction de fonction de gestion des outlier
    def imputOutlier(df, var):
        Q1 =df[var].quantile(0.25)
        Q3 =df[var].quantile(0.75)

        IQR= Q3-Q1
        min = Q1 -1.5*IQR
        max = Q1 +1.5*IQR

        df.loc[(df[var]<min) , var ] = min
        df.loc[(df[var]>max) , var ] = max
        return df
    #geestion des outlier
    for var in df.select_dtypes("number"):
        df  =imputOutlier(df, var)
    return df





# chargement des donnees

df = wrangle("base.csv")
print(df.info())
import statsmodels.formula.api as smf
import statsmodels.api as sm

# Liste des colonnes catégorielles
categorical_columns = df.select_dtypes(include=['category']).columns.tolist()

# Liste pour stocker les variables significatives
significant_variables = []
non_significatif = []
# Effectuer une ANOVA pour chaque variable catégorielle contre 'rendement'
for categorical_var in categorical_columns:
    # Vérifier le nombre de catégories uniques
    unique_values = df[categorical_var].nunique()
    if unique_values > 1:
        formula = f'rendement ~ C({categorical_var})'
        model = smf.ols(formula, data=df).fit()
        anova_table = sm.stats.anova_lm(model, typ=2)
        p_value = anova_table["PR(>F)"][0]

        print(f"ANOVA pour {categorical_var}:\n{anova_table}\n")

        if p_value < 0.05:
            print(f"La variable '{categorical_var}' a un effet significatif sur 'rendement' (p-value = {p_value:.4f}).\n")
            significant_variables.append(categorical_var)
        else:
            print(f"La variable '{categorical_var}' n'a pas d'effet significatif sur 'rendement' (p-value = {p_value:.4f}).\n")
            non_significatif.append(categorical_var)
    else:
        print(f"La variable '{categorical_var}' a moins de deux catégories uniques et est donc ignorée pour l'ANOVA.\n")

# Afficher les variables significatives
print("Variables significatives :", non_significatif)
print(len(significant_variables))

target ='rendement'
x= df.drop(columns=[target]+non_significatif)
y = df[target]
#separation de la base
x_train, x_test, y_train, y_test =tts(x,y,
                                    test_size=0.2,
                                    random_state = 3)
# Initialiser le scaler
scaler = StandardScaler()

# Fit and transform on the training data
x_train[x.select_dtypes(include="number").columns] = scaler.fit_transform(x_train[x.select_dtypes(include="number").columns])

# Use the same scaler to transform the test data (avoid data leakage)
x_test[x.select_dtypes(include="number").columns] = scaler.transform(x_test[x.select_dtypes(include="number").columns])


# Meilleurs paramètres trouvés
best_params = {'bootstrap': True, 'max_depth': 18, 'min_samples_leaf': 26, 'min_samples_split': 2, 'n_estimators': 50,'random_state': 42}

# Instancier le modèle avec les meilleurs paramètres
best_model = RandomForestRegressor(**best_params)

# Mesurer le temps d'entraînement
debut = time()
best_model.fit(x_train, y_train)
fin = time()

# Scores
score_train = best_model.score(x_train, y_train)
score_test = best_model.score(x_test, y_test)

# Préparer les résultats dans un DataFrame
results = pd.DataFrame({
    'modele': ['RandomForestRegressor'],
    'temps': [fin - debut],
    'score_train': [score_train],
    'score_test': [score_test]
})

# Afficher les résultats
print(results)
from sklearn.model_selection import cross_val_score
import numpy as np

# Instancier le modèle avec les meilleurs paramètres
best_model = RandomForestRegressor(**best_params)

# Mesurer le temps de validation croisée
debut = time()
cross_val_scores = cross_val_score(best_model, x_train, y_train, cv=5)  # 5-fold cross-validation
fin = time()

# Calculer les scores de validation croisée
mean_cross_val_score = np.mean(cross_val_scores)
std_cross_val_score = np.std(cross_val_scores)

# Scores sur l'ensemble de test
best_model.fit(x_train, y_train)
score_test = best_model.score(x_test, y_test)

# Préparer les résultats dans un DataFrame
results = pd.DataFrame({
    'modele': ['RandomForestRegressor'],
    'temps_validation_croisee': [fin - debut],
    'moyenne_score_val': [mean_cross_val_score],
    'ecart_type_score_val': [std_cross_val_score],
    'score_test': [score_test]
})

# Afficher les résultats
print(results)
@st.cache_resource
def get_model_and_scaler():
    scaler = StandardScaler()
    best_params = {
        'bootstrap': True, 'max_depth': 18, 'min_samples_leaf': 26,
        'min_samples_split': 2, 'n_estimators': 50, 'random_state': 42
    }
    model = RandomForestRegressor(**best_params)
    return scaler, model


def main():
    st.title('Application de Prédiction des Rendements Agricoles')

    # Barre latérale
    st.sidebar.title('Navigation')
    option = st.sidebar.radio("Choisissez une section", 
                              ["Home", 
                               "Prédiction"])

    # Charger les données depuis le fichier local
    filepath = "base.csv"  # Change ce chemin selon la localisation de ton fichier
    df = wrangle(filepath)

    # Définir la cible et les variables explicatives
    target = 'rendement'
    non_significatif = []  # Assurez-vous que cette liste est remplie correctement
    x = df.drop(columns=[target] + non_significatif)
    y = df[target]

    numeric_vars = ['superficie_parcelle', 'quantite_semence_CP', 'quantite_CP']  # Ajuster en fonction de tes variables numériques
    categorical_vars = ['id_reg',
    'culture_principale',
    'irrigation',
    'rotation_cultures',
    'culture_principale_précédente_2020_2021',
    'méthode_de_culture',
    'mois_semis_CP',
    'semaine_semis_CP',
    'etat_produit_CP',
    'mois_previsionnel_recolte_CP',
    'utilisation_matiere_org_avant_semis',
    'utilisation_matiere_miner_avant_semis',
    'utilisation_produits_phytosanitaires',
    'utilisation_residus_alimentation_animaux',
    'type_labour',
    'type_couverture_sol_interculture',
    'type_installation',
    'type_materiel_preparation_sol',
    'type_materiel_semis',
    'type_materiel_entretien_sol',
    'type_materiel_récolte',
    'production_forestiere_exploitation',
    'pratique_cultures_fourrageres']

    if option == "Home":
        # Charger l'image localement
        image_path = 'JWUFQ-8s.jpeg'  # Remplacez par le chemin correct
        image = Image.open(image_path)

        # Convertir l'image en base64 pour l'intégrer dans le HTML
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        # CSS pour styliser la page
        st.markdown("""
            <style>
            .main-content {
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
                background-color: #1e1e1e;
                color: #ffffff;
            }
            .text-content {
                flex: 1;
                padding-right: 20px;
            }
            .image-content {
                flex: 1;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            .main-content h1 {
                font-size: 36px;
                margin-bottom: 20px;
            }
            .main-content p {
                font-size: 18px;
                text-align: justify;
            }
            .main-content img {
                max-width: 100%;
                height: auto;
                border-radius: 10px;
                width: 100%;
            }
            </style>
        """, unsafe_allow_html=True)

        # Contenu principal
        st.markdown(f"""
        <div class="main-content">
            <div class="text-content">
                <h1>Prévision du rendement des cultures</h1>
                <p>
                    Nous avons développé des techniques efficaces 
                    pour estimer le rendement d'une culture à l'aide de l'apprentissage automatique. 
                    Nous utilisons les données d'exploitations agricoles issues de l'ANSD (Agence nationale de la Statistique et de la demographie) et de la DAPSA.
                </p>
            </div>
            <div class="image-content">
                <img src="data:image/jpeg;base64,{img_str}" alt="Prévision du rendement des cultures">
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.subheader('À propos du Projet')
        st.markdown("""
        Cette application a pour but de prédire les rendements agricoles en fonction de diverses variables influençant le rendement, telles que la superficie des parcelles, la quantité de semences utilisées, les pratiques culturales, et bien d'autres facteurs. 
        """)

        # Ajouter une section de chiffres clés
        st.subheader("Prévision du rendement des cultures en chiffres")

        # Custom CSS pour les boxs
        st.markdown("""
            <style>
            .box {
                padding: 20px;
                border-radius: 10px;
                background-color: #1e1e1e;
                box-shadow: 2px 2px 15px rgba(0, 0, 0, 0.3);
                text-align: center;
                margin-bottom: 20px;
                height: 350px; /* Hauteur fixe pour uniformiser les boxs */
            }
            .box h3 {
                color: #00cc66;
                font-size: 28px;
                margin: 0;
            }
            .box p {
                font-size: 16px;
                color: #dddddd;
                margin-top: 10px;
                text-align: justify; /* Justifier le texte */
            }
            </style>
            
        """, unsafe_allow_html=True)
            # Utilisation de colonnes pour afficher les chiffres clés dans des boxes
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="box">
                <h3>jusqu'à 75%</h3>
                <p>Précision de l'estimation des rendements dépend de la qualité des données statistiques et peut varier de 70 % à 75 %.</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="box">
                <h3>Jusqu'à 3 mois à l'a</h3>
                <p>Prédiction des rendements pour la saison donnée en cours jusqu'à 3 mois à l'avance.</p>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="box">
                <h3>50 +</h3>
                <p>Rendement prévu pour plusieurs types de cultures.</p>
            </div>
            """, unsafe_allow_html=True)

        col4, col5, col6 = st.columns(3)
        with col4:
            st.markdown("""
            <div class="box">
                <h3>Score du modele</h3>
                <p>Nous faisons une prévision des rendements avec  une précision de 75 % </p>
            </div>
            """, unsafe_allow_html=True)

        with col5:
            st.markdown("""
            <div class="box">
                <h3>0 à 100 champs</h3>
                <p>Modèle d'estimation des rendements </p>
            </div>
            """, unsafe_allow_html=True)

        with col6:
            st.markdown("""
            <div class="box">
                <h3>10 +</h3>
                <p>Nous nous assurons que les prévisions sont basées sur l'analyse de données la plus complète.</p>
            </div>
            """, unsafe_allow_html=True)

        st.subheader('MÉTHODOLOGIE')
    
        st.write("Notre Approche")
        st.write("Afin d'obtenir l'efficacité et la précision maximales des prévisions du rendement des cultures, nous avons entraine plusieurs modeles de prediction et choisi le meilleur pour garantir une prediction de qualite")
        
        # Ajouter deux nouvelles colonnes avec des boxes
        col7, col8 = st.columns(2)
        with col7:
            st.markdown("""
            <div class="box">
                <h3>Prévision des rendements:</h3>
                <p> 
                    Collecte des données (paramètres  analyse du sol, état de la culture etc.).<br>
                    Calibration du modèle et des hyperparametres pour assurer l'exactitude de la prévision du rendement des cultures en l'absence de données statistiques et pour augmenter la variabilité des valeurs.
                </p>
            </div>
            """, unsafe_allow_html=True)

        with col8:
            st.markdown("""
            <div class="box">
                <h3>Prévision des rendements</h3>
                                <p>Collecter des données pour créer l'ensemble de données de prévision du rendement des cultures et le combiner avec des prédicteurs possibles (pluies, température, humidité, type de sol, etc.).<br>
                Choix du bon modèle ML pour le projet, par exemple Random Forest, LightGBM, XGBoost, CatBoost, pour n'en nommer que quelques-uns.<br>
                </p>
             </div>
            """, unsafe_allow_html=True)

    elif option == "Prédiction":
        # Séparer les données
        x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=3)
        
        # Obtenir le scaler et le modèle mis en cache
        scaler, best_model = get_model_and_scaler()

        # Fit and transform on the training data
        x_train[numeric_vars] = scaler.fit_transform(x_train[numeric_vars])

        # Use the same scaler to transform the test data (avoid data leakage)
        x_test[numeric_vars] = scaler.transform(x_test[numeric_vars])
        
        # Mesurer le temps d'entraînement
        debut = time()
        best_model.fit(x_train, y_train)
        fin = time()

        # Scores
        score_train = best_model.score(x_train, y_train)
        score_test = best_model.score(x_test, y_test)

        # Préparer les résultats dans un DataFrame
        results = pd.DataFrame({
            'modele': ['RandomForestRegressor'],
            'temps': [fin - debut],
            'score_train': [score_train],
            'score_test': [score_test]
        })

        # Afficher les résultats
        st.write("Évaluation du Modèle")
        st.write(results)

        # Entrer les valeurs pour la prédiction
        st.subheader('Prédiction des Rendements')
        input_data = {}
        
        categorical_mappings = {
        'id_reg' :{1: 'DAKAR',
            2: 'ZIGUINCHOR',
            3: 'DIOURBEL',
            4: 'SAINT-LOUIS',
            5: 'TAMBACOUNDA',
            6: 'KAOLACK',
            7: 'THIES',
            8: 'LOUGA',
            9: 'FATICK',
            10: 'KOLDA',
            11: 'MATAM',
            12: 'KAFFRINE',
            13: 'KEDOUGOU',
            14: 'SEDHIOU'} ,
        
        'culture_principale': {
            1: 'Arachide',
            2: 'Aubergine',
            3: 'Béréf',
            4: 'Bissap',
            5: 'Coton',
            6: 'Diakhatou',
            7: 'Fonio',
            8: 'Gombo',
            9: 'Mais',
            10: 'Manioc',
            11: 'Mil',
            12: 'Niébé',
            13: 'Pastèque',
            14: 'Patate Douce',
            15: 'Piment',
            16: 'Riz irrigué'
         },
        
        'irrigation': {1:'Oui',2:'Non'},
       
        'rotation_cultures':{1:'Oui',2:'Non'},
        'culture_principale_précédente_2020_2021': {
            0: 'Nonexploitée',
            1: 'Arachide',
            2: 'Aubergine',
            3: 'Béréf',
            4: 'Bissap',
            5: 'Coton',
            6: 'Diakhatou',
            7: 'Fonio',
            8: 'Gombo',
            9: 'Mais',
            10: 'Manioc',
            11: 'Mil',
            12: 'Niébé',
            13: 'Pastèque',
            14: 'Patate Douce',
            15: 'Piment',
            16: 'Riz irrigué',
            17:'Rizpluvial',
            18:'Sésame',
            19:'Sorgho',
            20:'Tomate cérise',
            21:'Tomate industrielle',
            22:'Vouandzou', 
            23:'Choux', 
            24:'Oignon',
            25:'Pomme de terre',
            26:'Carotte',
            27:'Courge', 
            28:'Haricot vert',
            29:'Melon', 
            30:'Concombre',
            31:'Oignon vert', 
            32:'Nave'
         },
        'méthode_de_culture':{1:'Labour profond',2:'Labour peu profond',3:'Aucun labour'},
      
        'mois_semis_CP':{5:'mai',6:'juin',7:'juillet'},
        'semaine_semis_CP':{1:'Semaine 1',2:'Semaine 2',3:'Semaine 3',4:'Semaine 4'},
   
        'etat_produit_CP':{1:'epis',2:'graine',3:'non-decortique',4:'fruit/legume',5:'calice seche',6:'feuille',7:'fibre de coton',8:'coton graine',9:'tubercule',10:'gousse'},
        'mois_previsionnel_recolte_CP':{1:'janvier',2:'fevrier',3:'mars',4:'avril',5:'mais',6:'juin',7:'juillet',8:'aout',9:'septembre',10:'octobre',11:'novembre',12:'decembre'},
        'utilisation_matiere_org_avant_semis':{1:'Oui',2:'Non'},
        'utilisation_matiere_miner_avant_semis':{1:'Oui',2:'Non'},
        'utilisation_produits_phytosanitaires':{1:'Oui',2:'Non'},
        'utilisation_residus_alimentation_animaux':{1:'Oui',2:'Non'},
        'type_labour':{1:'Labour profond',2:'Labour peu profond',3:'Aucun labour'},
        'type_couverture_sol_interculture':{1:'Solnu/Pasdecouverture',2:'Résidusdeplantes'},
        'type_installation':{1:'Digues/Diguettes',2:'Cordons pierreux',3:'Canaux de drainage',4:'Brise vent et haies',5:'Gabion',6:'Aucuneinstallation'},
        'type_materiel_preparation_sol':{1:'Manuel',2:'Attele',3:'Motorise'},
        'type_materiel_semis':{1:'Manuel',2:'Attele',3:'Motorise'},
        'type_materiel_entretien_sol':{1:'Manuel',2:'Attele',3:'Motorise'},
        'type_materiel_récolte':{1:'Manuel',2:'Attele',3:'Motorise'},
        'production_forestiere_exploitation':{1:'Oui',2:'Non'},
        'pratique_cultures_fourrageres':{1:'Oui',2:'Non'}
            
       
        }
        
    
        # Variables numériques
        for var in numeric_vars:
            min_val, max_val = df[var].min(), df[var].max()
            input_data[var] = st.slider(f'Valeur pour {var}', min_value=float(min_val), max_value=float(max_val), value=float((min_val + max_val) / 2))

        # Variables catégorielles
        for var in categorical_vars:
            if var in categorical_mappings:
                options = categorical_mappings[var]
                reversed_options = {v: k for k, v in options.items()}
                selected_value = st.selectbox(f'Sélectionnez la catégorie pour {var}', list(options.values()))
                input_data[var] = reversed_options[selected_value]
            else:
                categories = df[var].unique()
                selected_category = st.selectbox(f'Sélectionnez la catégorie pour {var}', categories)
                input_data[var] = selected_category

        # Créer un DataFrame avec les inputs
        input_df = pd.DataFrame([input_data])

        # Ajouter les colonnes manquantes avec des valeurs par défaut
        for col in x.columns:
            if col not in input_df.columns:
                input_df[col] = np.nan

        # S'assurer que les colonnes sont dans le même ordre que x
        input_df = input_df[x.columns]

        # Normaliser les données numériques
        scaled_input_df = input_df.copy()
        scaled_input_df[numeric_vars] = scaler.transform(input_df[numeric_vars].fillna(0))

        # Faire la prédiction
        prediction = best_model.predict(scaled_input_df)
        st.write(f'Prédiction du rendement : {prediction[0]:.2f}')

if __name__ == "__main__":
    main()
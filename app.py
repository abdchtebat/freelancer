import pickle
import streamlit as st
import base64
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

if os.path.exists(csv_path):
    ht = pd.read_csv(csv_path)
else:
    st.error(f"Le fichier CSV n'a pas été trouvé à l'emplacement {csv_path}")









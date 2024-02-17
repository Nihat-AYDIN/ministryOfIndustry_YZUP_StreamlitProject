import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.svm import SVC
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import time

class App:
    def __init__(self):
        self.data = None
        self.dataset_name = None
        self.classifier_name = None
        self.X, self.y = None, None
        self.Init_Streamlit_Page()

    def run(self):
        self.get_dataset()
        self.PlotCorellationHeatMap()
        self.drawThePlot()
        # self.add_parameter_ui()
        # self.generate()

    def Init_Streamlit_Page(self):
        st.write("""
        # Exploring Dataset
        """)

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer',)
        )
        st.write(f"## {self.dataset_name} Dataset")

        self.classifier_name = st.sidebar.selectbox(
            'Select Classifier',
            ('KNN', 'SVM', 'Naïve Bayes')
        )

    def get_dataset(self):
        self.data = pd.read_csv("data.csv")
        self.y = self.data["diagnosis"] 
        self.X = self.data.drop("diagnosis", axis=1)
        st.write('Shape of dataset:', self.X.shape)
        st.write('Number of classes in the target column:', len(np.unique(self.y)))

        st.write('Feature columns: ', self.X.columns)
        st.markdown(f"<p>Target column: <span style='color: red; '><strong>  {self.y.name}</strong></span></p>", unsafe_allow_html=True)

        arr = self.data["diagnosis"].unique()
        st.markdown(f"<p>Name of the classes:  <span style='color: red; '><strong>  {arr[0]} ,{arr[1]}</strong></span></p>", unsafe_allow_html=True)

        st.write("## First 10 Row of The Data")
        st.write(self.data.head(10))
        st.write("## Last 10 Row of The Data")
        st.write(self.data.tail(10))

    def PlotCorellationHeatMap(self):
        st.write("## Before Removing High Correleted Columns :")
        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(self.data.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=.5)
        st.pyplot(heatmap.figure, use_container_width=True)

        # Korelasyon matrisini hesapla
        c = self.data.corr(numeric_only=True).abs()

        # Korelasyon matrisinin yalnızca üst üçgenini seç
        upper_triangle = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))

        # Üst üçgenin %90'dan fazla olan değerleri filtrele
        filtered_corr = upper_triangle[upper_triangle > 0.9].stack()
        
        st.dataframe(c.style.background_gradient(cmap ='coolwarm'))

        filtered_corr_df = pd.DataFrame(filtered_corr, columns=['corr_rate'])
        
        st.write("### Columns With a Correlation Ratio Greater Than 0.90:")
        st.write(filtered_corr_df)

        drops = self.remove_collinear_features(self.data, 0.90)
        self.data = self.data.drop(columns=drops)

        st.write("## After Removing High Correleted Columns:")
        plt.figure(figsize=(12, 10))
        heatmap_ = sns.heatmap(self.data.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=.5)
        st.pyplot(heatmap_.figure, use_container_width=True)

    def remove_collinear_features(self, dataframe, threshold):
        # Calculate the correlation matrix
        corr_matrix = dataframe.corr().abs()
        iters = range(len(corr_matrix.columns) - 1)
        drop_cols = []

        # Iterate through the correlation matrix and compare correlations
        for i in iters:
            for j in range(i+1):
                item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
                col = item.columns
                row = item.index
                val = abs(item.values)

                # If correlation exceeds the threshold
                if val >= threshold:
                    drop_cols.append(col.values[0])

        # Drop one of each pair of correlated columns
        drops = set(drop_cols)
        return drops

    def drawThePlot(self):
        # Renkleri belirle
        palette = sns.color_palette("husl", n_colors=len(self.data['diagnosis'].unique()))
        colors = dict(zip(self.data['diagnosis'].unique(), palette))

        # Etiketleri özelleştirerek Scatter plot çizimi
        fig, ax = plt.subplots()
        for label, color in colors.items():
            subset = self.data[self.data['diagnosis'] == label]
            if label == 'M':
                label_text = 'Malignant'
            elif label == 'B':
                label_text = 'Benign'
            else:
                label_text = label
            ax.scatter(subset['radius_mean'], subset['texture_mean'], c=color, label=f"{label_text}")

        ax.set_xlabel('Radius Mean')
        ax.set_ylabel('Texture Mean')
        ax.legend()

        st.write('## Scatter Plot of Radius Mean vs Texture Mean: ')
        st.pyplot(fig)


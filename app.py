import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

class App:
    def __init__(self):
        self.data = None
        self.dataset_name = None
        self.classifier_name = None
        self.X, self.y = None, None
        self.cv = None

        self.X_train, self.X_test, \
        self.y_train, self.y_test = None, None, None, None

        self.params = dict()
        self.clf = None

        self.Content_Table()

    def run(self):
        self.Init_Streamlit_Page()
        self.get_dataset()
        self.add_parameter_ui()
        self.PlotCorellationHeatMap()
        self.drawThePlot()
        self.Getting_Ready_the_Data()
        self.Model()
        # self.add_parameter_ui()
        # self.generate()

    def Content_Table(self):
        st.markdown("<h1 style='color: #9AD0C2; text-align: center;'>Table of Contents</h1>", unsafe_allow_html=True)

        ingredients = ["Exploring Dataset", "Data Preprocessing"]
        # ingredients başlıklarını tanımla ve renk kodunu ayarla
        style = "style='color: #2D9596;'"
        titles = [f"<h5 {style}>{title}</h5>" for title in ingredients]

        # İçindekiler listesini oluştur
        contents_list = ["- [{}](#{})".format(title, title.replace(" ", "-").lower()) for title in ingredients]

        # Başlıkları göster
        for title in titles:
            st.markdown(title, unsafe_allow_html=True)

    def Init_Streamlit_Page(self):
        st.markdown("<h1 style='color: #720455; text-align: center;'>Exploring Dataset</h1>", unsafe_allow_html=True)

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer',)
        )
        st.markdown(f"<h3 style='color: #F6B17A;'>{self.dataset_name} Dataset</h3>", unsafe_allow_html=True)
        # st.write(f"## {self.dataset_name} Dataset")

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
        st.markdown(f"<p>Target column: <span style='color: #910A67; '><strong>  {self.y.name}</strong></span></p>", unsafe_allow_html=True)

        arr = self.data["diagnosis"].unique()
        st.markdown(f"<p>Name of the classes:  <span style='color: #910A67; '><strong>  {arr[0]} ,{arr[1]}</strong></span></p>", unsafe_allow_html=True)

        st.markdown(f"<h3 style='color: #F6B17A;'>First 10 Row of The Data</h3>", unsafe_allow_html=True)
        st.write(self.data.head(10))
        st.markdown(f"<h3 style='color: #F6B17A;'>Last 10 Row of The Data</h3>", unsafe_allow_html=True)
        st.write(self.data.tail(10))

    def PlotCorellationHeatMap(self):
        st.markdown("<h1 style='color: #720455; text-align: center;'>Data Preprocessing</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #F6B17A;'>Before Removing High Correleted Columns :</h3>", unsafe_allow_html=True)

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
        
        st.markdown(f"<h5 style='color: #F9F7C9;'>Columns With a Correlation Ratio Greater Than 0.90 :</h5>", unsafe_allow_html=True)

        st.write(filtered_corr_df)

        drops = self.remove_collinear_features(self.data, 0.90)
        drops_df = pd.DataFrame(drops, columns=['will_be_deleted'])
        st.markdown(f"<h5 style='color: #F9F7C9;'>Columns to be deleted due to high correlation :</h5>", unsafe_allow_html=True)
        st.write(drops_df)
        self.data = self.data.drop(columns=drops)

        st.markdown(f"<h3 style='color: #F6B17A;'>After Removing High Correleted Columns :</h3>", unsafe_allow_html=True)
        plt.figure(figsize=(12, 10))
        heatmap_ = sns.heatmap(self.data.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=.5)
        st.pyplot(heatmap_.figure, use_container_width=True)
    
    def Getting_Ready_the_Data(self):
        # Change the M value in the 'diagnosis' column to 1 and the B value to 0
        self.data['diagnosis'] = self.data['diagnosis'].map({'M': 1, 'B': 0})

        # Split the data 80-20 percent into X_train, Y_train, X_test, and Y_test.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)


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
            ax.scatter(subset['radius_mean'], subset['texture_mean'], c=color, label=f"{label_text}", alpha=0.5)

        ax.set_xlabel('Radius Mean')
        ax.set_ylabel('Texture Mean')
        ax.legend()

        st.markdown(f"<h3 style='color: #F6B17A;'>Scatter Plot of Radius Mean vs Texture Mean :</h3>", unsafe_allow_html=True)
        st.pyplot(fig)

    def add_parameter_ui(self):
        if self.classifier_name != "Naïve Bayes":
            self.cv = st.sidebar.slider("Cross-validation count", min_value=2, max_value=5, step=1, value=3)
        if self.classifier_name == 'SVM':
            self.params['c1'] = st.sidebar.slider('c1', min_value=0.1, max_value=3.0, step=0.1, value=0.1)
            self.params['c2'] = st.sidebar.slider('c2', min_value=3.01, max_value=5.0, step=0.1, value=3.0)
        elif self.classifier_name == 'KNN':
            K1 = st.sidebar.slider('k1', min_value=1, max_value=3, step=1, value=2)
            K2 = st.sidebar.slider('k2', min_value=4, max_value=7, step=1, value=6)
            K3 = st.sidebar.slider('k3', min_value=8, max_value=10, step=1, value=9)
            self.params['n_neighbors'] = [K1, K2, K3]
            self.params['weights'] = ['uniform', 'distance']

    def Model(self):
        if self.classifier_name == 'SVM':
            self.clf  = SVC(C=self.params['C'])
        elif self.classifier_name == 'KNN':
            knn = KNeighborsClassifier()

            grid_search = GridSearchCV(knn, self.params, cv=self.cv)
            grid_search.fit(self.X_train, self.y_train)
    
            st.markdown(f"<h5 style='color: #F9F7C9;'>The best combination of parameters for {self.classifier_name} :</h5>",
             unsafe_allow_html=True)
            st.write(grid_search.best_params_)

            st.markdown(f"<h5 style='color: #F9F7C9;'>Evaluate the test set using the best model</h5>", unsafe_allow_html=True)
            # Predictions on the test set
            y_pred = grid_search.predict(self.X_test)

            # Classification report
            report_str = classification_report(self.y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report_str)
            st.write(report_df)

            st.markdown(f"<h5 style='color: #F9F7C9;'>Confusion Matrix :</h5>", unsafe_allow_html=True)
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)

            # Plot confusion matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='inferno', cbar=False, 
                        xticklabels=['Predicted B', 'Predicted M'],
                        yticklabels=['Actual B', 'Actual M'])
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)

            # Accuracy
            accuracy = grid_search.score(self.X_test, self.y_test)
            st.write("Test set accuracy score:", accuracy)
        else:
            self.clf  = RandomForestClassifier(n_estimators=self.params['n_estimators'],
                max_depth=self.params['max_depth'], random_state=1234)

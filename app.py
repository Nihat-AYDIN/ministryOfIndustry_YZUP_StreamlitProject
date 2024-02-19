import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_option('deprecation.showPyplotGlobalUse', False)

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

        self.Content_Table()

    def run(self):
        self.Init_Streamlit_Page()
        self.get_Dataset()
        self.plot_Corellation_HeatMap()
        self.draw_the_Excpected_Plot()
        self.about_Kaggle_Research()
        self.getting_Ready_the_Data()
        self.add_Parameter()
        self.Model()

    def Content_Table(self):
        st.markdown("<h1 style='color: #9AD0C2; text-align: center;'>Table of Contents</h1>", unsafe_allow_html=True)

        # Define the headers of ingredients and set the color code.
        ingredients = ["Exploring Dataset", "Data Preprocessing", "Research on the Dataset", "Building Model"]
        style = "style='color: #2D9596;'"
        titles = [f"<h5 {style}>{title}</h5>" for title in ingredients]
        contents_list = ["- [{}](#{})".format(title, title.replace(" ", "-").lower()) for title in ingredients]

        # Show the headers.
        for title in titles:
            st.markdown(title, unsafe_allow_html=True)

    def Init_Streamlit_Page(self):
        st.markdown("<h1 style='color: #720455; text-align: center;'>Exploring Dataset</h1>", unsafe_allow_html=True)

        self.dataset_name = st.sidebar.selectbox(
            'Select Dataset',
            ('Breast Cancer',)
        )
        st.markdown(f"<h3 style='color: #F6B17A;'>{self.dataset_name} Dataset</h3>", unsafe_allow_html=True)

        self.classifier_name = st.sidebar.selectbox(
            'Select Classifier',
            ('KNN', 'SVM', 'Naïve Bayes')
        )

    def get_Dataset(self):
        self.data = pd.read_csv("data.csv")
        self.y = self.data["diagnosis"] 
        self.X = self.data.drop("diagnosis", axis=1)
        st.write('Shape of dataset:', self.X.shape)
        st.write('Number of classes in the target column:', len(np.unique(self.y)))

        st.write('Feature columns: ', self.X.columns)
        st.markdown(f"<p>Target column: <span style='color: #910A67; '><strong>  {self.y.name}</strong></span></p>", unsafe_allow_html=True)

        arr = self.data["diagnosis"].unique()
        st.markdown(f"<p>Name of the classes:  <span style='color: #910A67; '><strong>  {arr[0]} ,{arr[1]}</strong></span></p>", unsafe_allow_html=True)

        dfM = self.data[self.data["diagnosis"] == "M"] 
        dfB = self.data[self.data["diagnosis"] == "B"] 

        st.write('The number of rows with the target value "M":', dfM.shape[0])
        st.write('The number of rows with the target value "B":', dfB.shape[0])

        st.markdown(f"<h3 style='color: #F6B17A;'>First 10 Row of The Data</h3>", unsafe_allow_html=True)
        st.write(self.data.head(10))
        st.markdown(f"<h3 style='color: #F6B17A;'>Last 10 Row of The Data</h3>", unsafe_allow_html=True)
        st.write(self.data.tail(10))

    def plot_Corellation_HeatMap(self):
        st.markdown("<h1 style='color: #720455; text-align: center;'>Data Preprocessing</h1>", unsafe_allow_html=True)
        st.markdown(f"<h3 style='color: #F6B17A;'>Before Removing High Correleted Columns :</h3>", unsafe_allow_html=True)

        plt.figure(figsize=(12, 10))
        heatmap = sns.heatmap(self.data.corr(numeric_only=True), annot=False, cmap='coolwarm', linewidths=.5)
        st.pyplot(heatmap.figure, use_container_width=True)

        # Calculate the correlation matrix.
        c = self.data.corr(numeric_only=True).abs()

        # Select only the upper triangle of the correlation matrix.
        upper_triangle = c.where(np.triu(np.ones(c.shape), k=1).astype(bool))

        # Filter values in the upper triangle that are more than 90%.
        filtered_corr = upper_triangle[upper_triangle > 0.9].stack()
        
        st.dataframe(c.style.background_gradient(cmap ='coolwarm'))

        filtered_corr_df = pd.DataFrame(filtered_corr, columns=['corr_rate'])
        
        st.markdown(f"<h5 style='color: #F9F7C9;'>Columns With a Correlation Ratio Greater Than 0.90 :</h5>", unsafe_allow_html=True)

        st.write(filtered_corr_df)

    def about_Kaggle_Research(self):
        st.markdown(f"<h1 style='color: #720455; text-align: center;'>Research on the Dataset</h1>", unsafe_allow_html=True)
        st.write("After making some observations regarding the dataset, I looked into some Kaggle projects to gain a better understanding. In one study I found, I learned that feature selection was applied, revealing that certain columns had a more significant impact on the models.")
        st.link_button("The Kaggle notebook I received help from.", "https://www.kaggle.com/code/mirichoi0218/classification-breast-cancer-or-not-with-15-ml#conclusion")
        st.markdown(f"<h5 style='color: #F9F7C9;'>Impactful Column :</h5>",
        unsafe_allow_html=True)

        useful_column = {
            "columns": ["radius_mean",
                        "texture_mean",
                        "smoothness_mean",
                        "compactness_mean", 
                        "concavity_mean", 
                        "perimeter_mean", 
                        "area_mean", 
                        "symmetry_mean",
                        "concave points_mean"]
        }
        df = pd.DataFrame(useful_column)
        st.write(df)

        st.pyplot(self.plot())

    
    def plot(self):
        # generate a scatter plot matrix with the "mean" columns
        cols = ["diagnosis", 
                "radius_mean",
                "texture_mean",
                "smoothness_mean",
                "compactness_mean", 
                "concavity_mean", 
                "perimeter_mean", 
                "area_mean", 
                "symmetry_mean",
                "concave points_mean"]
        return sns.pairplot(data=self.data[cols], hue='diagnosis', palette='rocket')
 
    def getting_Ready_the_Data(self):

        self.data = self.data.drop("id", axis=1)
        self.data["diagnosis"] = self.data["diagnosis"].map({'M': 1, 'B': 0})
        
        indices = []
        num_cols = self.data.columns
        for col in num_cols:
            indices.extend(self.diffrent_outlier(self.data, col, "diagnosis", std1=3.5, std2=3.5))
        indices = set(indices)

        st.markdown(f"<h3 style='color: #F6B17A;'>Outlier Detection :</h3>", unsafe_allow_html=True)
        st.markdown(f"<h5 style='color: #F9F7C9;'>Let's Perform Outlier Detection Using Standard Deviation :</h5>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #F9F7C9;'>I found {len(indices)} outliers in the data.</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color: #F9F7C9;'>I tried several values for the threshold of standard deviation, but through my experiments, \
                     I found that the most optimal value was 3.5</p>", unsafe_allow_html=True)

        st.markdown(f"<h5 style='color: #F9F7C9;'>Let's visualize Outliers in a Plot :</h5>", unsafe_allow_html=True)

        plt.figure(figsize=(10, 8))

        # Scatter plot for non-outliers
        sns.scatterplot(data=self.data[~self.data.index.isin(indices)], x="radius_mean", y="texture_mean", hue="diagnosis", palette=['green', 'blue'], alpha=0.7)

        # Scatter plot for outliers with a different color
        sns.scatterplot(data=self.data[self.data.index.isin(indices)], x="radius_mean", y="texture_mean", color='red', s=100)

        plt.title("Scatter Plot of radius_mean vs texture_mean")
        plt.xlabel("radius_mean")
        plt.ylabel("texture_mean")
        plt.legend()
                
        st.pyplot()

        # Let's drop the indices we have obtained from the data.
        self.data = self.data.drop(indices, axis=0)

        self.X = self.data[["radius_mean",
                            "texture_mean",
                            "smoothness_mean",
                            "compactness_mean", 
                            "concavity_mean", 
                            "perimeter_mean", 
                            "area_mean", 
                            "symmetry_mean",
                            "concave points_mean"]]
        self.y = self.data["diagnosis"]

        # Split the data 80-20 percent into X_train, Y_train, X_test, and Y_test.
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=23)

    def diffrent_outlier(self, df, num_col_name, target, std1=3, std2=3):
        """
        This function sequentially takes numeric columns in a for loop and performs an outlier scan based on the class names in the target.
        The method used here is based on standard deviation.
        It returns the indices of outlier values as a list.
        """
        outlier_indices = []

        for class_ in df[target].unique():
            selected_class = df[df[target] == class_]
            selected_col = selected_class[num_col_name]

            std = selected_col.std()
            avg = selected_col.mean()
            
            three_sigma_plus = avg + (std1 * std)
            three_sigma_minus = avg - (std2 * std)

            outlier_count = (selected_class[num_col_name] > three_sigma_plus).sum() + (selected_class[num_col_name]< three_sigma_minus).sum()
            outliers = selected_class[(selected_col > three_sigma_plus) | (selected_col < three_sigma_minus)]
            outlier_indices.extend(outliers.index.tolist())

        return outlier_indices

    def draw_the_Excpected_Plot(self):
        # Renkleri belirle
        palette = sns.color_palette("husl", n_colors=len(self.data['diagnosis'].unique()))
        colors = dict(zip(self.data['diagnosis'].unique(), palette))

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
    
    def add_Parameter(self):
        if self.classifier_name != "Naïve Bayes":
            self.cv = st.sidebar.slider("Cross-validation count", min_value=2, max_value=5, step=1, value=3)
        if self.classifier_name == 'SVM':
            self.params = { 'C':[0.001, 0.01, 0.1],
                            'kernel':['poly', 'rbf', 'sigmoid']
                        }

        elif self.classifier_name == 'KNN':
            self.params = { 'n_neighbors':np.arange(1,50),
                            'p':np.arange(1,3),
                            'weights':['uniform','distance'],
                            'algorithm':['auto','ball_tree']
                        }

    def Model(self):
        st.markdown(f"<h1 style='color: #720455; text-align: center;'>Building Model </h1>",
        unsafe_allow_html=True)
        if self.classifier_name == 'SVM':
            grid_search = GridSearchCV(SVC(), self.params, cv=self.cv)
            grid_search.fit(self.X_train, self.y_train)
    
            st.markdown(f"<h5 style='color: #F9F7C9;'>The parameters used for tuning {self.classifier_name} :</h5>",
             unsafe_allow_html=True)
            st.write(self.params)

            st.markdown(f"<h5 style='color: #F9F7C9;'>The best combination of parameters for {self.classifier_name} :</h5>",
             unsafe_allow_html=True)
            st.write(grid_search.best_params_)

            st.markdown(f"<h5 style='color: #F9F7C9;'>Evaluate the test set using the best model</h5>", unsafe_allow_html=True)
            # Predictions on the test set
            y_pred = grid_search.predict(self.X_test)

        elif self.classifier_name == 'KNN':
            grid_search = GridSearchCV(KNeighborsClassifier(), self.params, cv=self.cv)
            grid_search.fit(self.X_train, self.y_train)

            st.markdown(f"<h5 style='color: #F9F7C9;'>The parameters used for tuning {self.classifier_name} :</h5>",
             unsafe_allow_html=True)
            st.write(self.params)
    
            st.markdown(f"<h5 style='color: #F9F7C9;'>The best combination of parameters for {self.classifier_name} :</h5>",
             unsafe_allow_html=True)
            st.write(grid_search.best_params_)

            st.markdown(f"<h5 style='color: #F9F7C9;'>Evaluate the test set using the best model</h5>", unsafe_allow_html=True)
            # Predictions on the test set
            y_pred = grid_search.predict(self.X_test)

        else:
            model = MultinomialNB()
            model.fit(self.X_train, self.y_train)
            
            st.markdown(f"<h5 style='color: #F9F7C9;'>Result for {self.classifier_name} :</h5>",
             unsafe_allow_html=True)

            # Test seti üzerinde tahminlerde bulunalım
            y_pred = model.predict(self.X_test)

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


## About the Project
Crime prediction is crucial for law enforcement and public safety. Recent advancements in AI offer new opportunities for reliable crime prediction models. This project aims to create a comprehensive crime prediction model for South Australia, focusing on deterrent practices. Using the Random Forest algorithm, the model identifies intricate patterns in crime data. It offers practical insights for crime prevention and promotes evidence-based decision-making by incorporating anti-crime characteristics like community policing programs and law enforcement tactics.
The project aims to develop a precise crime prediction model using advanced machine learning methods like Random Forest, Gaussian Naive Bayes, Ridge Regression, Linear Support Vector Classification, and K-Nearest Neighbours. The model will use historical data and anti-crime features to forecast crime rates. The research will provide evidence-based insights for law enforcement, policymakers, and community stakeholders, facilitating targeted resource allocation and crime prevention strategies.


## Getting Started
Our team members have looked through Australia's crime data and for reliable data sources for forecasting. The website for South Australia contained files that were best for us to use among the sites that we looked at. We came across different online data varying into various format. The collection includes crucial data such as the reported incident dates, in-depth descriptions of the offences, and numerical counts of these offences. A careful analysis of this collected data was conducted to identify the most significant features that contribute to crime prediction.

### Prerequisites
Our project is using Visual Studio as the source code platform since it has all those relevant packages that can be used to develop our code more efficiently. The basic requirements would be installing Visual studio and a system with 32gb RAM. The source code can also be executed on Google Colab since it is a well-suited platform for data analysis. It should have a sufficient number of compute units and a fast GPU.. There are certain important library files that needs to be imported which are all related to data analysis and visualization. Below are all the specific packages installed before running the project:

1. pandas (imported as pd): Used for data manipulation and analysis.
2. numpy (imported as np): Provides support for numerical operations and arrays.
3. seaborn (imported as sns): Used for statistical data visualization, particularly informative and attractive statistical graphics.
4. sklearn.model_selection: Contains functions for splitting data into training and testing sets and performing cross-validation.
5. sklearn.tree: Includes Decision Tree related models and functions.
6. graphviz: Provides tools for creating and rendering graph visualizations.
7. IPython.display: Used for displaying images within the IPython environment.
8. sklearn.naive_bayes: Contains functions for implementing the Naive Bayes classification algorithm.
9. sklearn.feature_selection: Contains functions for feature selection, such as SelectKBest and chi2 for choosing the most important features.
10. matplotlib.pyplot (imported as plt): A library for creating static, animated, and interactive visualizations in Python.
11. sklearn.svm: Contains functions related to Support Vector Machines.
12. sklearn.linear_model: Contains functions for linear regression and related models.
13. sklearn.preprocessing: Provides functions for data preprocessing, including scaling and transforming data.
14. sklearn.decomposition: Contains functions for matrix factorization and dimensionality reduction techniques like PCA.
15. sklearn.neighbors: Contains functions for k-nearest neighbors regression.
16. sklearn.metrics: Contains functions for evaluating the performance of machine learning models, such as mean_absolute_error, confusion_matrix, and classification_report.
17. yellowbrick.regressor: Contains functions for visualizing regression models, such as ResidualsPlot.

### Installation
Our project requires installing pip and python on the system. Below are the important pip packages that needs to be installed:
1. pip install pandas: This package helps in reading the data from the various file formats in data frames. It manages the missing values and removing the outliers.
2. pip install numpy: This mainly supports all the numerical operations and the operations involving solving of arrays.
3. pip install seaborn: This supports in creating graphical representations.
4. pip install sci-kit learn: This includes machine learning models like classification and regression techniques. It also helps in model training and validation. 
5. pip install graphviz: This is used for creating the visualization charts.
6. pip install matplotlib: Supports in creating different types of plots.
7. pip install yellowbrick: Supports model evaluation and feature importance. 

## Data

Different Government websites offered crime-based statistics at local, state, and national levels. These websites provided up-to-date information on crimes and additional details, encouraging transparency and accessibility to official government data. Various data repositories have been found to be helpful for our model. The South Australian crime dataset is the foundation of our analysis, which was meticulously prepared through a thorough data cleaning process, encoding categorical variables, and structuring dates. Data aggregation at various levels improved the predictive models’ adaptability and accuracy. Below is the online source from which we collected our data. 
https://data.sa.gov.au/data/dataset/crime-statistics
Our data set consists of the following information which is used for producing the classification and the regression techniques:
1. Reported Data: The date the crime was reported ranging from the year 2010-2023.
2. Suburb-Incident: The suburb name where the crime was reported.
3. Offence Level 1 description: Describes the offence type
4. Offence Level 2 description: Detail on the offence that was raised.
5. Offence Level 3 description: Specifies the category under which the offence would be counted.
6. Offence count: The count of the offence that took place on that suburb.

We included an anti-crime element to the algorithm to improve the models’ predicted efficacy and accuracy. A dataset of people detained on remand at specific locations was obtained. Following that, this dataset was incorporated with the dataset on crime statistics to provide our model with further training and testing data. Hence, we merged the above data to an additional information which was taken from the below source:
https://www.abs.gov.au/statistics/people/crime-and-justice/prisoners-australia/latest-release#state-territory 

## Exploratory Data Analysis
The proposed methodology for the crime prediction project involves a systematic approach to gather and analyse data, develop predictive models, and incorporate anti-crime actions.
The following steps outline the methodology followed:
Data Collection: The primary data source for this project is from the South Australian government’s open data portal. The collection includes crucial data such as the reported incident dates, in-depth descriptions of the offences, and numerical counts of these offences. The project focuses on SA as the geographical area of interest to analyse and predict crime patterns accurately.
Data Pre-processing: The collected crime data underwent a thorough pre-processing stage. This step involved cleaning the data, handling missing values, and transforming the data into a suitable format for analysis. Features such as crime type, location, time, and additional relevant variables are extracted and processed to capture meaningful patterns and trends.
Feature Engineering: Domain knowledge and insights from crime experts was incorporated to engineer informative features that capture important aspects of crime behaviour and demographics specific to South Australia.
Feature Selection: A careful analysis of the collected data was conducted to identify the most significant features that contribute to crime prediction. Feature selection techniques, such as correlation analysis and statistical tests, was used to identify relevant predictors.
Model Development: The models were trained on the historical crime data, using appropriate evaluation metrics to assess their performance. Techniques such as cross-validation and hyperparameter tuning was employed to optimize the models.



## Modeling
Our project encompasses both regression and classification tasks, each employing distinct machine learning techniques.
Regression: 
1. Principal Component Analysis (PCA): We applied PCA to reduce dimensionality, enhancing model computational efficiency.
2. Ridge Regression: Utilizing historical data and engineered features, we employed Ridge Regression to predict crime counts.
3. K-Nearest Neighbours (KNN) Regression: KNN regression was employed to make predictions based on the similarity of crime patterns in the dataset.

Classification
1. Random Forest Classifier: This ensemble learning technique was deployed to classify criminal offences, offering high accuracy and robustness.
2. Support Vector Classifier (SVC): SVC was utilized for binary and multiclass classification tasks, capable of handling complex decision boundaries.
3. Gaussian Naive Bayes: We explored the Gaussian Naive Bayes classifier’s simplicity and efficiency for classification tasks, assuming normal feature distributions.

## Evaluation

We utilized a suite of evaluation metrics tailored to the specific task, including Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), F1-score, and accuracy. Cross-validation techniques ensured our models’ generalization to unseen data. Our project focuses on predicting crime rates in South Australia by harnessing machine learning techniques. With a solid foundation in data pre-processing and a diverse set of predictive models, we aim to provide law enforcement agencies with valuable insights to enhance public safety and resource allocation strategies. Our detailed model evaluations and results contribute to the selection of the most suitable predictive models for crime prediction in South Australia. This project represents a significant step toward leveraging data and machine learning to improve crime prevention and intervention efforts in the region.

## Results
We generated a classification report. This report includes metrics like precision, recall, F1-score, and support for each class. It provides insights into the model’s performance on different classes.
 
![classificationReport1](https://github.com/AdibaHasin/Crime-Prediction-Capston-Unit-/assets/44343038/e5dfc4d8-e725-4567-8756-4cb2356447fa)
![ClassificationReport2](https://github.com/AdibaHasin/Crime-Prediction-Capston-Unit-/assets/44343038/f090feef-6eba-4102-89a7-93af4b8c3cda)
![classificatioReport3](https://github.com/AdibaHasin/Crime-Prediction-Capston-Unit-/assets/44343038/3ac3d3ad-fc9a-4385-a8a9-98c08a6a203b)


## Acknowledgements
We extend our gratitude to the entire team for their collaborative efforts and dedication in contributing to all aspects of this project. The shared commitment and mutual support among team members have been instrumental in fostering our collective growth and learning throughout this project. Special thanks to our faculty members Janusz Getta and Fuchun Guo for their valuable insights and assistance in developing our project, which significantly enriched the outcomes.

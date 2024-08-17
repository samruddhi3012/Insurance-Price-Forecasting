# Insurance Price Forecast 

## 🕵️ Objectives
The objective of this project is to analyse the insurance dataset and build an optimized machine learning model to make accurate predictions of insurance costs based on the provided dataset. 

## 💻 Data
The dataset contains historical records of 1338 insured customers with the following columns:
* Age: Age of the primary beneficiary.
* Sex: Gender of the primary beneficiary.
* BMI: Body mass index of the primary beneficiary.
* Children: Number of children the primary beneficiary has.
* Smoker: Whether the primary beneficiary smokes.
* Region: The primary beneficiary's residential area in the US.
* Charges: Individual medical costs billed by health insurance.
  
## 🪛 Tools Used

* Tools : Python, Jupyter Notebook
  
* Keywords: Linear Regression, XGBoost, Box-Cox, Recursive Feature Elimination (RFE), BayesSearchCV, Sklearn Pipeline, Statistical Analysis, Chi-Squared test, ANOVA test, Exploratory data analysis, Univariate and Bivariate analysis, Quantile Quantile Plot, Trend Analysis
  
* Libraries: Pandas, Numpy, Scipy, Scikit-learn, Xgboost, Skopt
  <ul>
  <li>Scipy</li>
  
       pip install scipy
  </ul>
  <ul>
  <li>Scikit-learn</li>
  
       pip install -U scikit-learn
  </ul>
  <ul>
  <li>Skopt</li>
  
       pip install scikit-optimize
  </ul>
  
## 🔖 Results
<!--* EDA
    1. After performing the EDA (from correlation section) it can be concluded that the children column has a low correlation to the target (charges).
    2. On the other hand for categorical features, from chi-squared test it is observed that sex and smoker have correlated to each other.
    3. From ANOVA test, it is seen that the column region is correlated to bmi.
    4. The features sex and region have low correlation with target (charges).
   --> 
* Baseline Linear Model
    1. Transformation: Used Box-Cox for data transformation.
    2. Training RMSE: 5830.10
    3. R-Squared: 0.80
* XGBoost model
   1. Optimization: Used Sklearn's Pipeline and Skopt's BayesSearchCV.
   2. Training RMSE: 4627.11
   3. R-Squared: 0.87
      
* Comparison of models.
  1. From the baseline to the xgb model, there was a **20.63%** improvement in the RMSE value for the test set.
  2. The R-Squared value for test set has improved by **8.75%** from baseline to xgb model.
     
## 📝 Description 

* _**Data Preparation**_:
  1. Imported essential libraries for data manipulation and visualization.
  2. Cleaned the data to handle missing values, correct data types, and remove inconsistencies.
     
* _**Exploratory Data Analysis (EDA)**_:
<br>EDA was conducted to understand the dataset's underlying patterns and relationships. This section covered:
  * Distribution Analysis
  * Univariate Data Analysis (wrt target)
  * Bivariate Data Analysis (wrt target)
  * Collinearity between Features
  * Correlation between Features

* _**Build and evaluate a baseline linear model**_:
<br> A baseline model was built using Linear Regression. This section covered:
  * Data Transformation
  * Understanding Linear Regression Assumptions
  * Building Linear Regression
  * Validating Linear Regression Assumptions
  * Model Training
  * Model Evaluation: <br>Assessing the model's performance using metrics like R-squared and Mean Squared Error (MSE).

* _**Model Building using XGBoost Regressor**_:
<br> An advanced model was built using the XGBoost Regressor. This section included:
  * Preparing data specifically for the XGBoost model.
  * Building Pipelines with Sklearn’s Pipeline Operator
  * Implementing BayesSearchCV for XGBoost Hyperparameter Optimization: <br> Using Bayesian optimization to fine-tune the hyperparameters of the XGBoost model.
  * Model Evaluation
    
* _**Comparison of Models**_:
<br> The performance of the Linear Regression model and the XGBoost Regressor was compared. The primary evaluation metric used for comparison was the Root Mean Square Error (RMSE).

* _**Performance of the Models**_:
<br>The final step involved a detailed analysis of the models' performance. The XGBoost Regressor, with optimized hyperparameters, was expected to outperform the baseline Linear Regression model in terms of predictive accuracy and RMSE.

<!--
# Introduction
Insurance companies cover several health and property related costs. Medical expenditures, house and car damage, fire incidents, and financial losses can all be included among these outlays. Traditionally, computing insurance rates requires a lot of manual labor and effort to fit the ever more complicated data.

Insurance firms must have a consistent method to forecast healthcare costs and guarantee their profitability if they are to survive. Our aim is to design a machine learning model that forecasts the charges or payments made by the health insurance company, therefore guiding the rates and maintaining profitability.

-->
<!--

* _**Exploratory Data Analysis (EDA)**_:
<br>EDA was conducted to understand the dataset's underlying patterns and relationships. This section covered:
  * Distribution Analysis: <br> Analyzing the distribution of features to understand their spread and central tendency.
  * Univariate Data Analysis (wrt target):<br>Examining individual features in relation to the target variable to identify significant predictors.
  * Bivariate Data Analysis (wrt target): <br>Assessing relationships between pairs of features and the target variable, for both numeric and categorical features.
  * Collinearity between Features:
    1. For numeric columns, a correlation matrix was plotted to identify multicollinearity.
    2. For categorical features, the chi-squared test was performed.
    3. For collinearity between numeric and categorical features, ANOVA tests were conducted.
  * Correlation:
    1. For numeric columns, a correlation matrix was plotted.
    2. For categorical features, ANOVA tests were performed to assess their relationship with the target variable.

* _**Build and evaluate a baseline linear model**_:
<br> A baseline model was built using Linear Regression. This section covered:
  * OneHot Encoding for Categorical Variables
  * Splitting Data
  * Data Transformation: <br>Scaling and normalizing data to meet model assumptions.
  * Understanding Linear Regression Assumptions: <br>Ensuring the assumptions of linear regression were met.
  * Building Linear Regression
  * Validating Linear Regression Assumptions:<br> Checking for linearity, independence, homoscedasticity, and normality of residuals.
  * Model Training
  * Model Evaluation: <br>Assessing the model's performance using metrics like R-squared and Mean Squared Error (MSE).
  * Residuals: <br>Analyzing residuals to ensure the model's accuracy.
  * Homoscedasticity: <br>Checking for constant variance in residuals.

  -->
<!--
# Detailed Description

3. Exploratory Data Analysis (EDA)
EDA was conducted to understand the dataset's underlying patterns and relationships. This section covered:

Distribution Analysis: Analyzing the distribution of features to understand their spread and central tendency.
Univariate Data Analysis (wrt target): Examining individual features in relation to the target variable to identify significant predictors.
Bivariate Data Analysis (wrt target): Assessing relationships between pairs of features and the target variable, for both numeric and categorical features.
Collinearity between Features:
For numeric columns, a correlation matrix was plotted to identify multicollinearity.
For categorical features, the chi-squared test was performed.
For collinearity between numeric and categorical features, ANOVA tests were conducted.
Correlation:
For numeric columns, a correlation matrix was plotted.
For categorical features, ANOVA tests were performed to assess their relationship with the target variable.
4. Baseline Model Building
A baseline model was built using Linear Regression. This section covered:

OneHot Encoding for Categorical Variables: Converting categorical variables into a numerical format using one-hot encoding.
Splitting Data: Dividing the dataset into training and testing sets.
Data Transformation: Scaling and normalizing data to meet model assumptions.
Understanding Linear Regression Assumptions: Ensuring the assumptions of linear regression were met.
Implementing Linear Regression: Building the linear regression model.
Validating Linear Regression Assumptions: Checking for linearity, independence, homoscedasticity, and normality of residuals.
Model Training: Training the model on the training data.
Model Evaluation: Assessing the model's performance using metrics like R-squared and Mean Squared Error (MSE).
Residuals: Analyzing residuals to ensure the model's accuracy.
Homoscedasticity: Checking for constant variance in residuals.
5. Model Building using XGBoost Regressor
An advanced model was built using the XGBoost Regressor. This section included:

Data Processing: Preparing data specifically for the XGBoost model.
Building Pipelines with Sklearn’s Pipeline Operator: Creating streamlined workflows for data preprocessing and model training.
Implementing BayesSearchCV for XGBoost Hyperparameter Optimization: Using Bayesian optimization to fine-tune the hyperparameters of the XGBoost model.
Model Evaluation: Evaluating the model's performance using appropriate metrics.
-->

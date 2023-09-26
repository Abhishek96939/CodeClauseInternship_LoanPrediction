# CodeClauseInternship_LoanPrediction
The provided Python code is a machine learning script that performs classification on a dataset for loan approval prediction. Here's a summary of the key steps and actions in the code:

1. **Data Import and Exploration:**
   - The code starts by importing necessary libraries, such as NumPy, Pandas, and Matplotlib.
   - It reads a dataset named 'Loan_Data.csv' and displays its first few rows, info, and statistical summary.

2. **Data Preprocessing:**
   - Missing values in the dataset are handled:
     - Numerical columns are interpolated to fill missing values.
     - Categorical columns (e.g., Gender, Married) are filled with the most frequent category.
   - Label encoding is applied to convert categorical variables into numerical format.

3. **Train-Test Split:**
   - The dataset is split into training and testing sets using a 75-25% ratio.

4. **Feature Scaling:**
   - Standardization is performed on the feature data using StandardScaler to ensure all features have the same scale.

5. **Model Building and Evaluation:**
   - Five different machine learning models are trained and evaluated on the data.
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Naive Bayes
     - K-Nearest Neighbors (KNN)
     - Support Vector Machine (SVM)
   - For each model, the following steps are executed:
     - The model is trained on the training data.
     - Predictions are made on the test data.
     - A confusion matrix and accuracy score are computed to evaluate model performance.
     - Confusion matrices are visualized using seaborn's heatmap.

6. **Conclusion:**
   - The script concludes by indicating which model performed the best based on accuracy scores and confusion matrices. In this case, Logistic Regression is chosen as the best-performing model for the task of loan approval prediction.

This code provides a comprehensive overview of how to preprocess data, train multiple machine learning models, and evaluate their performance for a binary classification task. The choice of the best model depends on the dataset and the specific problem being addressed.

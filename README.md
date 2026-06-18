# Student Performance and Placement Prediction Using Machine Learning

A complete AI & Machine Learning Mini Project in Python designed to analyze student academic performance and behavioral patterns. The project utilizes supervised learning to predict placement outcomes and unsupervised learning to segment students for proactive academic counseling.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Workflow Explanation](#workflow-explanation)
3. [Required Libraries](#required-libraries)
4. [Folder Structure](#folder-structure)
5. [How to Run in VS Code](#how-to-run-in-vs-code)
6. [Expected Outputs](#expected-outputs)
7. [Viva Questions & Answers](#viva-questions--answers)

---

## Project Overview

* **Goal**: To build a model that predicts whether a student will be placed or not based on academic and behavioral metrics (CGPA, attendance, study hours, project score, assignment score, internal marks, and communication skills). Additionally, to segment students into three distinct cohorts (High Performers, Average Performers, and At-Risk Students) using clustering.
* **Supervised Learning Model**: Random Forest Classifier
* **Unsupervised Learning Model**: K-Means Clustering ($K=3$)
* **Technologies Used**: Python, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn.

---

## Workflow Explanation

The project is structured into 8 distinct phases:

### Phase 1: Dataset Understanding
* **Action**: Loads `student_placement.csv` using Pandas.
* **Analysis**: Displays the first 5 records, dataset dimensions (`shape`), column names, variable data types (`dtypes`), missing values count per column, and standard statistical summaries (mean, min, max, std).
* **Explanation**: Prints descriptive interpretations to explain what the data reveals at first glance.

### Phase 2: Data Preprocessing
* **Action**: Cleans and prepares the data for machine learning.
* **Steps**:
  1. Identifies and removes duplicate records.
  2. Imputes missing numerical values (e.g., in `CGPA` and `Attendance`) using columns' median values to handle missingness.
  3. Uses Scikit-learn's `LabelEncoder` to transform categorical text columns (`Communication_Skills` and target variable `Placement_Status`) into numerical format.
  4. Separates independent features ($X$) from the target column ($y$, `Placement_Status`).
  5. Scales numerical columns using `StandardScaler` to ensure mean = 0 and standard deviation = 1 (critical for clustering and model convergence).
  6. Splits the data into an 80% training set and a 20% testing set using stratified sampling to maintain class ratios.

### Phase 3: Exploratory Data Analysis (EDA)
* **Action**: Generates and saves high-quality Matplotlib & Seaborn visualizations to the `visualizations/` directory:
  * **Chart 1: Placement Distribution**: A pie chart showing the percentage of placed vs. unplaced students.
  * **Chart 2: Attendance vs Placement**: A boxplot illustrating how class attendance levels differ between placed and unplaced students.
  * **Chart 3: Study Hours vs CGPA**: A scatterplot with a regression trendline depicting the relationship between daily study hours and cumulative CGPA.
  * **Chart 4: Feature Importance**: (Generated after Phase 4) A horizontal bar chart ranking student attributes by their relative predictive weight in the Random Forest.

### Phase 4: Supervised Learning Model
* **Action**: Trains a `RandomForestClassifier` on the training dataset and predicts placement outcomes on the test set.
* **Output**: Displays sample test cases, comparing the ground-truth placement status against the model's prediction.
* **Explanation**: Explains the rationale for selecting Random Forest (ensemble learning, non-linear relationships, robustness to outliers, built-in feature importance).

### Phase 5: Model Evaluation
* **Action**: Assesses prediction capabilities using standard metrics.
* **Metrics**: Generates a Confusion Matrix, Accuracy, Precision, Recall, and F1-Score.
* **Formulas Included in Comments**:
  * $\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}$
  * $\text{Precision} = \frac{TP}{TP + FP}$
  * $\text{Recall} = \frac{TP}{TP + FN}$
  * $\text{F1-Score} = \frac{2 \times \text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$

### Phase 6: Unsupervised Learning Model
* **Action**: Selects academic and behavioral features and applies `KMeans` clustering with $K = 3$.
* **Mapping**: Group labels (0, 1, 2) are mapped dynamically by sorting CGPAs:
  * **High Performers**: Cluster with the highest mean CGPA.
  * **Average Performers**: Cluster with the intermediate mean CGPA.
  * **At-Risk Students**: Cluster with the lowest mean CGPA.
* **Visualization**: Saves a cluster scatter plot (`chart5_kmeans_clusters.png`) of Attendance vs. CGPA, highlighting cluster centroids with an 'X' marker.

### Phase 7: Findings
* **Action**: Prints high-level conclusions drawn from feature importances, data distributions, and cluster characteristics.

### Phase 8: Conclusion
* **Action**: Summarizes project outcomes, performance metrics, and the practical utility of applying this model in schools and universities.

---

## Required Libraries

Ensure you have the following packages installed:
* `numpy` (Numerical vector operations)
* `pandas` (Dataframe loading and manipulation)
* `matplotlib` (Static charting engine)
* `seaborn` (Statistical visualization enhancement)
* `scikit-learn` (Machine learning classifiers, metrics, and preprocessing)

To install them, run the following command in your terminal:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---

## Folder Structure

```
student-placement-project/
│
├── student_placement_prediction.py     # Main Python project source code
├── generate_data.py                    # Helper data generator (run first if CSV missing)
├── student_placement.csv               # Dataset containing student records
├── README.md                           # Project documentation (this file)
│
└── visualizations/                     # Folder containing saved project plots
    ├── chart1_placement_distribution.png
    ├── chart2_attendance_vs_placement.png
    ├── chart3_study_hours_vs_cgpa.png
    ├── chart4_feature_importance.png
    └── chart5_kmeans_clusters.png
```

---

## How to Run in VS Code

1. **Open Workspace**: Open VS Code, select **File > Open Folder...**, and select the project folder containing the files.
2. **Open Terminal**: Go to **Terminal > New Terminal** (or press \`Ctrl + \` \`).
3. **Verify Dataset**: If `student_placement.csv` is not present, running `student_placement_prediction.py` will automatically generate it.
4. **Execute Python File**:
   * Type the following command in the terminal and press Enter:
     ```bash
     python student_placement_prediction.py
     ```
   * Alternatively, click the **Play** button (Run Python File) in the top-right corner of the editor.
5. **View Outputs**:
   * Review text summaries printed directly in the terminal console.
   * Open the newly generated `visualizations/` folder in the VS Code sidebar to view the charts.

---

## Expected Outputs

### Console Log (Summary)
* **Dataset Shape**: `(200, 9)`
* **Data Preprocessing**: Shows that duplicates are removed, nulls in `CGPA` and `Attendance` are replaced, and text values are encoded.
* **Sample Predictions**: A table displaying actual vs. predicted values (e.g., Placed vs. Not Placed) with a Boolean match column.
* **Supervised Metrics**:
  * Accuracy: $\approx 92\% - 97\%$
  * Precision: $\approx 90\% - 96\%$
  * Recall: $\approx 90\% - 96\%$
  * F1-Score: $\approx 90\% - 96\%$
* **Unsupervised Cluster Profiles**: Detailed statistics for the three student cohorts.

---

## Viva Questions & Answers

### 1. What is Random Forest Classifier, and why is it called an "Ensemble" method?
* **Answer**: Random Forest is a supervised learning algorithm that builds a forest of multiple Decision Trees. It is called an ensemble method because it combines predictions from several base estimators (Decision Trees) to make a final prediction (using majority voting for classification). Ensemble methods reduce model variance, making them more robust and less prone to overfitting than individual decision trees.

### 2. Explain K-Means Clustering. What does the parameter 'K' stand for?
* **Answer**: K-Means is an unsupervised clustering algorithm that partitions data into $K$ distinct, non-overlapping groups (clusters). The letter $K$ represents the user-defined number of clusters. The algorithm works by:
  1. Randomly placing $K$ centroids.
  2. Assigning each data point to its nearest centroid (using Euclidean distance).
  3. Recalculating centroids by averaging the points in each cluster.
  4. Repeating steps 2 and 3 until centroids stabilize.

### 3. What is the difference between Precision and Recall?
* **Answer**:
  * **Precision** measures the quality of positive predictions: out of all students predicted as "Placed", how many actually got placed? Formula: $\frac{TP}{TP + FP}$.
  * **Recall** (Sensitivity) measures the model's ability to find all positive cases: out of all students who actually got placed, how many did the model identify? Formula: $\frac{TP}{TP + FN}$.

### 4. Why do we need the F1-Score? Why not rely only on Accuracy?
* **Answer**: Accuracy can be misleading if the dataset classes are highly imbalanced (e.g., 95% placed, 5% not placed). A dummy model predicting "Placed" for everyone would achieve 95% accuracy but fails completely at detecting unplaced students. The **F1-Score** is the harmonic mean of Precision and Recall. It provides a balanced score that requires both precision and recall to be high, making it a much better metric for evaluating performance on skewed datasets.

### 5. Why is Standard Scaling (StandardScaler) applied before K-Means Clustering?
* **Answer**: K-Means relies heavily on Euclidean distance to determine similarity. If features have vastly different scales (e.g., CGPA ranges from 2.0 to 4.0, while Attendance ranges from 50 to 100), the larger-scale feature (Attendance) will dominate the distance calculations, making the CGPA irrelevant. `StandardScaler` centers the data (mean = 0, std = 1) so that all features contribute equally to clustering.

### 6. What is the role of LabelEncoder in the Preprocessing phase?
* **Answer**: Machine learning algorithms in scikit-learn require numerical inputs. Categorical columns like `Communication_Skills` (Low, Medium, High) and `Placement_Status` (Placed, Not Placed) contain text. `LabelEncoder` translates these text labels into ordinal integers (e.g., Low $\rightarrow$ 0, Medium $\rightarrow$ 1, High $\rightarrow$ 2), making them compatible with Scikit-learn's modeling functions.

### 7. How did you identify which K-Means cluster is "At-Risk" vs "High Performers"?
* **Answer**: Since K-Means assigns cluster labels (0, 1, 2) arbitrarily, we group the data by cluster and calculate the average `CGPA` for each cluster. The cluster with the highest average CGPA is labeled "High Performers", the cluster with the lowest average CGPA is labeled "At-Risk Students", and the intermediate cluster is labeled "Average Performers".

### 8. What are the key findings from this student placement project?
* **Answer**:
  1. Both academic results (CGPA) and behavioral metrics (Attendance, Study Hours) are heavily correlated with placement success.
  2. Study hours are the primary driver of high CGPAs.
  3. Feature importance highlights CGPA, Attendance, and Project Score as the most influential variables in predicting placement status.
  4. Students categorized in the "At-Risk" cluster show significantly lower placement rates, highlighting the need for early academic intervention.

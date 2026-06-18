import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans

# Set style for modern aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({
    'axes.edgecolor': '#cccccc', 'axes.linewidth': 0.8,
    'grid.color': '#eeeeee', 'grid.linestyle': '--',
    'figure.titlesize': 12, 'axes.titlesize': 11, 'figure.dpi': 120
})

def print_phase_header(num, title):
    print(f"\n{'='*80}\n{f' PHASE {num}: {title} '.center(80, '=')}\n{'='*80}")

def check_or_create_dataset(filename="student_placement.csv"):
    if os.path.exists(filename):
        return
    print(f"[INFO] Dataset '{filename}' not found. Generating synthetic dataset...")
    np.random.seed(42)
    n = 10000
    
    # Generate realistic correlated features
    att = np.random.uniform(55, 98, n)
    hours = np.random.uniform(1, 10, n)
    cgpa = np.clip(2.0 + hours * 0.15 + (att - 50) * 0.015 + np.random.normal(0, 0.15, n), 2.0, 4.0)
    ints = np.clip(40 + cgpa * 12 + np.random.normal(0, 5, n), 50, 100)
    assign = np.clip(35 + hours * 3 + (att - 50) * 0.8 + np.random.normal(0, 6, n), 50, 100)
    proj = np.clip(45 + cgpa * 10 + hours * 2 + np.random.normal(0, 5, n), 50, 100)
    
    # Communication skills
    comm_choices = ["Low", "Medium", "High"]
    comm = [np.random.choice(comm_choices, p=[0.1, 0.3, 0.6] if cgpa[i] > 3.4 else [0.5, 0.4, 0.1] if cgpa[i] < 2.6 else [0.3, 0.4, 0.3]) for i in range(n)]
    
    # Placement probability calculation
    comm_num = np.array([0 if c == "Low" else 1 if c == "Medium" else 2 for c in comm])
    prob = (cgpa - 2.0) * 2.5 + ((att - 60) / 10) * 1.2 + comm_num * 0.9 + np.random.normal(0, 1.0, n)
    placed = ["Placed" if p >= 4.0 else "Not Placed" for p in prob]
    
    df = pd.DataFrame({
        "Student_ID": [f"STU{i:05d}" for i in range(1, n + 1)],
        "Attendance": np.round(att, 2), "Study_Hours": np.round(hours, 2), "CGPA": np.round(cgpa, 2),
        "Internal_Marks": np.round(ints, 2), "Assignment_Score": np.round(assign, 2),
        "Communication_Skills": comm, "Project_Score": np.round(proj, 2), "Placement_Status": placed
    })
    
    # Inject missing values
    df.loc[np.random.choice(n, 3, replace=False), "CGPA"] = np.nan
    df.loc[np.random.choice(n, 3, replace=False), "Attendance"] = np.nan
    df.to_csv(filename, index=False)

# ==============================================================================
# PHASE 1: DATASET UNDERSTANDING
# ==============================================================================
def phase_1_dataset_understanding(filename):
    print_phase_header(1, "DATASET UNDERSTANDING")
    df = pd.read_csv(filename)
    print("1. First 5 Rows:\n", df.head())
    print(f"\n2. Shape: {df.shape}")
    print("\n3. Column Names:\n", list(df.columns))
    print("\n4. Data Types:\n", df.dtypes)
    print("\n5. Missing Values:\n", df.isnull().sum())
    print("\n6. Statistical Summary:\n", df.describe())
    return df

# ==============================================================================
# PHASE 2: DATA PREPROCESSING
# ==============================================================================
def phase_2_data_preprocessing(df):
    print_phase_header(2, "DATA PREPROCESSING")
    df_clean = df.copy().drop_duplicates()
    
    # Handle missing values
    imputed_info = []
    for col in df_clean.columns:
        if df_clean[col].isnull().any():
            fill_val = df_clean[col].median() if df_clean[col].dtype in ['float64', 'int64'] else df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(fill_val)
            imputed_info.append(f"Imputed '{col}' using median: {fill_val:.2f}")
            
    # Encode categorical columns
    encoders = {}
    encoded_info = []
    for col in ["Communication_Skills", "Placement_Status"]:
        le = LabelEncoder()
        df_clean[col] = le.fit_transform(df_clean[col])
        encoders[col] = le
        mapping_str = ", ".join([f"{c}->{i}" for i, c in enumerate(le.classes_)])
        encoded_info.append(f"'{col}' ({mapping_str})")
        
    X = df_clean.drop(columns=["Student_ID", "Placement_Status"])
    y = df_clean["Placement_Status"]
    
    # Scale numerical features
    num_cols = ["Attendance", "Study_Hours", "CGPA", "Internal_Marks", "Assignment_Score", "Project_Score"]
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[num_cols] = scaler.fit_transform(X[num_cols])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    
    # Print tabular preprocessing report
    print(f"| {'Preprocessing Step':22} | {'Status / Details':51} |")
    print("-" * 80)
    print(f"| {'1. Duplication Check':22} | {'Duplicates removed (Final rows: ' + str(df_clean.shape[0]) + ')':51} |")
    
    if imputed_info:
        for idx, info in enumerate(imputed_info):
            step_name = "2. Missing Values" if idx == 0 else ""
            print(f"| {step_name:22} | {info:51} |")
    else:
        print(f"| {'2. Missing Values':22} | {'No missing values detected.':51} |")
        
    for idx, info in enumerate(encoded_info):
        step_name = "3. Label Encoding" if idx == 0 else ""
        print(f"| {step_name:22} | {info:51} |")
        
    print(f"| {'4. Feature Scaling':22} | {'StandardScaler applied to ' + str(len(num_cols)) + ' numerical features':51} |")
    print(f"| {'5. Dataset Splitting':22} | {f'Train Set: {len(X_train)} rows | Test Set: {len(X_test)} rows':51} |")
    print("-" * 80)
    
    return X_train, X_test, y_train, y_test, scaler, encoders, df_clean

# ==============================================================================
# PHASE 3: EXPLORATORY DATA ANALYSIS
# ==============================================================================
def phase_3_exploratory_data_analysis(df_clean, encoders):
    print_phase_header(3, "EXPLORATORY DATA ANALYSIS")
    os.makedirs("visualizations", exist_ok=True)
    colors = ["#2ec4b6", "#e71d36"]
    
    df_plot = df_clean.copy()
    df_plot["Placement"] = encoders["Placement_Status"].inverse_transform(df_clean["Placement_Status"])
    
    # Chart 1: Placement Distribution
    fig, ax = plt.subplots(figsize=(5, 5))
    counts = df_clean["Placement_Status"].value_counts()
    ax.pie(counts, labels=encoders["Placement_Status"].inverse_transform(counts.index), autopct='%1.1f%%', colors=colors)
    ax.set_title("Placement Status Distribution", fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/chart1_placement_distribution.png", bbox_inches='tight')
    plt.close()

    # Chart 2: Attendance vs Placement
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(x="Placement", y="Attendance", data=df_plot, hue="Placement", palette=colors, ax=ax, width=0.5, legend=False)
    ax.set_title("Attendance vs Placement Status", fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/chart2_attendance_vs_placement.png", bbox_inches='tight')
    plt.close()

    # Chart 3: Study Hours vs CGPA
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(x="Study_Hours", y="CGPA", hue="Placement", data=df_plot, palette=colors, ax=ax)
    sns.regplot(x="Study_Hours", y="CGPA", data=df_plot, scatter=False, ax=ax, color="black", line_kws={"linestyle": "--"})
    ax.set_title("Study Hours vs CGPA", fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/chart3_study_hours_vs_cgpa.png", bbox_inches='tight')
    plt.close()
    print("Saved EDA charts to 'visualizations/' folder.")

# ==============================================================================
# PHASE 4: SUPERVISED LEARNING MODEL
# ============================================================================== 
def phase_4_supervised_learning(X_train, X_test, y_train, y_test, encoders):
    print_phase_header(4, "SUPERVISED LEARNING MODEL")
    print("Why Random Forest: Robust ensemble learning, handles mixed variable types, evaluates non-linear thresholds, extracts feature importance.")
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    
    print("\nSample Predictions (First 10):")
    act = encoders["Placement_Status"].inverse_transform(y_test[:10])
    pred = encoders["Placement_Status"].inverse_transform(y_pred[:10])
    for a, p in zip(act, pred):
        print(f"  Actual: {a:12} | Predicted: {p:12} | Match: {a==p}")
        
    return rf, y_pred

def plot_feature_importance(rf_model, feature_names):
    print("\nPlotting Feature Importance (Chart 4)...")
    importances = rf_model.feature_importances_
    indices = np.argsort(importances)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(range(len(importances)), importances[indices], color="#2ec4b6", align='center')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_title("Feature Importance", fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/chart4_feature_importance.png", bbox_inches='tight')
    plt.close()

# ==============================================================================
# PHASE 5: MODEL EVALUATION
# ==============================================================================
def phase_5_model_evaluation(y_test, y_pred, encoders):
    print_phase_header(5, "MODEL EVALUATION")
    
    # Formulas:
    # Accuracy = (TP + TN) / (TP + TN + FP + FN)
    # Precision = TP / (TP + FP)
    # Recall = TP / (TP + FN)
    # F1 Score = 2 * Precision * Recall / (Precision + Recall)
    
    cm = confusion_matrix(y_test, y_pred)
    classes = encoders["Placement_Status"].classes_
    print("Confusion Matrix:\n", pd.DataFrame(cm, index=[f"Actual {c}" for c in classes], columns=[f"Predicted {c}" for c in classes]))
    
    print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_test, y_pred):.4f}")

# ==============================================================================
# PHASE 6: UNSUPERVISED LEARNING MODEL
# ==============================================================================
def phase_6_unsupervised_learning(df_clean, encoders):
    print_phase_header(6, "UNSUPERVISED LEARNING MODEL")
    features = ["Attendance", "Study_Hours", "CGPA", "Internal_Marks", "Assignment_Score", "Project_Score"]
    
    X_clust = StandardScaler().fit_transform(df_clean[features])
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_clean["Cluster"] = kmeans.fit_predict(X_clust)
    
    # Dynamic group labeling by mean CGPA
    means = df_clean.groupby("Cluster")["CGPA"].mean().sort_values()
    name_map = {means.index[0]: "At-Risk Students", means.index[1]: "Average Performers", means.index[2]: "High Performers"}
    df_clean["Segment"] = df_clean["Cluster"].map(name_map)
    
    # Visualize K-Means
    fig, ax = plt.subplots(figsize=(6, 5))
    group_colors = {"High Performers": "#2ec4b6", "Average Performers": "#ff9f1c", "At-Risk Students": "#e71d36"}
    sns.scatterplot(x="Attendance", y="CGPA", hue="Segment", data=df_clean, palette=group_colors, ax=ax)
    
    for segment, color in group_colors.items():
        sub = df_clean[df_clean["Segment"] == segment]
        ax.plot(sub["Attendance"].mean(), sub["CGPA"].mean(), marker='X', color='black', markersize=12, markeredgecolor='white')
        
    ax.set_title("Student Segments (K-Means)", fontweight="bold")
    plt.tight_layout()
    plt.savefig("visualizations/chart5_kmeans_clusters.png", bbox_inches='tight')
    plt.close()
    
    print("Cluster Profiles:")
    print(f"{'Segment':20} | {'Count (N)':<9} | {'Avg CGPA':<8} | {'Avg Study (h)':<13} | {'Placement Rate':<14}")
    print("-" * 76)
    for group in ["High Performers", "Average Performers", "At-Risk Students"]:
        sub = df_clean[df_clean["Segment"] == group]
        print(f"{group:20} | {len(sub):<9} | {sub['CGPA'].mean():<8.2f} | {sub['Study_Hours'].mean():<13.2f} | {(sub['Placement_Status']==1).mean()*100:<13.1f}%")

# ==============================================================================
# MAIN SCRIPT EXECUTION
# ==============================================================================
def main():
    csv_file = "student_placement.csv"
    check_or_create_dataset(csv_file)
    
    df = phase_1_dataset_understanding(csv_file)
    X_train, X_test, y_train, y_test, scaler, encoders, df_clean = phase_2_data_preprocessing(df)
    phase_3_exploratory_data_analysis(df_clean, encoders)
    
    rf_model, y_pred = phase_4_supervised_learning(X_train, X_test, y_train, y_test, encoders)
    plot_feature_importance(rf_model, list(X_train.columns))
    
    phase_5_model_evaluation(y_test, y_pred, encoders)
    phase_6_unsupervised_learning(df_clean, encoders)

if __name__ == "__main__":
    main()

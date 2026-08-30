import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

def load_benchmark_data(name):
    """بارگذاری مستقیم دیتاست‌های بنچ‌مارک استاندارد مقاله از UCI"""
    if name == 'ecoli1':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
        df = pd.read_csv(url, sep=r'\s+', header=None)
        X = df.iloc[:, 1:8].values
        y = np.where(df.iloc[:, 8] == 'im', 1, 0)
        
    elif name == 'yeast1':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
        df = pd.read_csv(url, sep=r'\s+', header=None)
        X = df.iloc[:, 1:9].values
        y = np.where(df.iloc[:, 9] == 'NUC', 1, 0)
        
    elif name == 'haberman':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data"
        df = pd.read_csv(url, header=None)
        X = df.iloc[:, 0:3].values
        y = np.where(df.iloc[:, 3] == 2, 1, 0)  # فوت در ۵ سال = ۱ (کلاس اقلیت)
        
    elif name == 'glass1':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        df = pd.read_csv(url, header=None)
        X = df.iloc[:, 1:10].values
        y = np.where(df.iloc[:, 10] == 1, 1, 0) # شیشه ساختمان = ۱
        
    else:
        raise ValueError(f"Dataset {name} not found.")
        
    return X, y

def get_classifiers():
    return {
        'DT': DecisionTreeClassifier(random_state=42),
        'kNN': KNeighborsClassifier(n_neighbors=5),
        'LR': LogisticRegression(random_state=42),
        'SVM': SVC(probability=True, random_state=42),
        'RF': RandomForestClassifier(n_estimators=100, random_state=42)
    }

def run_reproduction_benchmark():
    datasets = ['ecoli1', 'yeast1', 'haberman', 'glass1']
    records = []
    
    print("======================================================================")
    print("STARTING FULL REPRODUCTION EXPERIMENT (ARTICLE PROTOCOL)")
    print("======================================================================")
    
    for ds_name in datasets:
        X, y = load_benchmark_data(ds_name)
        n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)
        print(f"\n--> Dataset: {ds_name:10s} | Shape: {X.shape} | Imbalance: {n_pos}:{n_neg} ({n_neg/n_pos:.2f}:1)")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        classifiers = get_classifiers()
        
        for clf_name, clf in classifiers.items():
            base_f1_list, froid_f1_list = [], []
            base_auc_list, froid_auc_list = [], []
            
            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                # پیش‌پردازش استاندارد بدون نشت داده
                scaler = StandardScaler()
                X_train_sc = scaler.fit_transform(X_train)
                X_test_sc = scaler.transform(X_test)
                
                # ۱. مدل پایه (Baseline)
                clf_base = clf.__class__(**clf.get_params())
                clf_base.fit(X_train_sc, y_train)
                p_base = clf_base.predict(X_test_sc)
                prob_base = clf_base.predict_proba(X_test_sc)[:, 1] if hasattr(clf_base, "predict_proba") else p_base
                
                base_f1_list.append(f1_score(y_test, p_base, pos_label=1, zero_division=0))
                base_auc_list.append(roc_auc_score(y_test, prob_base))
                
                # ۲. روش مقاله (FROID: PCA + Isolation Forest)
                n_comp = min(2, X_train.shape[1])
                pca = PCA(n_components=n_comp, random_state=42)
                Z_train = pca.fit_transform(X_train_sc)
                Z_test = pca.transform(X_test_sc)
                
                od = IsolationForest(random_state=42)
                od.fit(Z_train)
                od_train = -od.decision_function(Z_train).reshape(-1, 1)
                od_test = -od.decision_function(Z_test).reshape(-1, 1)
                
                X_train_froid = np.hstack([X_train_sc, od_train])
                X_test_froid = np.hstack([X_test_sc, od_test])
                
                clf_froid = clf.__class__(**clf.get_params())
                clf_froid.fit(X_train_froid, y_train)
                p_froid = clf_froid.predict(X_test_froid)
                prob_froid = clf_froid.predict_proba(X_test_froid)[:, 1] if hasattr(clf_froid, "predict_proba") else p_froid
                
                froid_f1_list.append(f1_score(y_test, p_froid, pos_label=1, zero_division=0))
                froid_auc_list.append(roc_auc_score(y_test, prob_froid))
                
            records.append({
                'Dataset': ds_name,
                'Classifier': clf_name,
                'Base F1': f"{np.mean(base_f1_list):.4f} ± {np.std(base_f1_list):.3f}",
                'FROID F1': f"{np.mean(froid_f1_list):.4f} ± {np.std(froid_f1_list):.3f}",
                'Delta F1 (%)': f"{((np.mean(froid_f1_list) - np.mean(base_f1_list)) / (np.mean(base_f1_list) + 1e-6) * 100):+.2f}%",
                'Base AUC': f"{np.mean(base_auc_list):.4f}",
                'FROID AUC': f"{np.mean(froid_auc_list):.4f}"
            })
            
    df_res = pd.DataFrame(records)
    print("\n" + "="*80)
    print("COMPREHENSIVE ARTICLE REPRODUCTION RESULTS")
    print("="*80)
    print(df_res.to_string(index=False))
    
    df_res.to_csv("full_reproduction_results.csv", index=False)
    print("\nSaved full results to full_reproduction_results.csv successfully.")

if __name__ == '__main__':
    run_reproduction_benchmark()

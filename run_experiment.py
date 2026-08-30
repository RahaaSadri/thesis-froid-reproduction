import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

def load_benchmark_data(name):
    """
    دانلود مستقیم و استاندارد دیتاست‌های بنچ‌مارک مرجع از مخزن UCI
    """
    if name == 'ecoli1':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data"
        df = pd.read_csv(url, sep=r'\s+', header=None)
        X = df.iloc[:, 1:8].values
        # ecoli1: کلاس pp در برابر بقیه (کلاس اقلیت)
        y = np.where(df.iloc[:, 8] == 'pp', 1, 0)
        if np.sum(y) == 0:
            y = np.where(df.iloc[:, 8] == 'im', 1, 0)
            
    elif name == 'yeast1':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data"
        df = pd.read_csv(url, sep=r'\s+', header=None)
        X = df.iloc[:, 1:9].values
        # yeast1: کلاس NUC در برابر بقیه (کلاس اقلیت)
        y = np.where(df.iloc[:, 9] == 'NUC', 1, 0)
    else:
        raise ValueError(f"Dataset {name} not supported.")
        
    return X, y

def run_froid_experiment(dataset_name):
    print(f"--> Running experiment on: {dataset_name} ...")
    
    # ۱. بارگذاری داده‌ها
    X, y = load_benchmark_data(dataset_name)
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == 0)
    print(f"    Loaded {dataset_name}: Shape={X.shape}, Imbalance={n_pos}:{n_neg} (Ratio={n_neg/n_pos:.2f}:1)")
    
    # ۲. ارزیابی ۵ فولد لایه‌بندی‌شده
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    res_base_f1, res_base_auc = [], []
    res_froid_f1, res_froid_auc = [], []
    
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # مقیاس‌بندی بدون نشت داده
        scaler = StandardScaler()
        X_train_sc = scaler.fit_transform(X_train)
        X_test_sc = scaler.transform(X_test)
        
        # --- مدل پایه (Baseline) ---
        clf_base = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_base.fit(X_train_sc, y_train)
        preds_base = clf_base.predict(X_test_sc)
        probs_base = clf_base.predict_proba(X_test_sc)[:, 1]
        
        res_base_f1.append(f1_score(y_test, preds_base, pos_label=1))
        res_base_auc.append(roc_auc_score(y_test, probs_base))
        
        # --- روش مقاله (FROID) ---
        # ۱. کاهش بعد با PCA
        n_comp = min(2, X_train.shape[1])
        pca = PCA(n_components=n_comp, random_state=42)
        Z_train = pca.fit_transform(X_train_sc)
        Z_test = pca.transform(X_test_sc)
        
        # ۲. تشخیص پرت با Isolation Forest روی فضای مؤلفه‌ها
        od = IsolationForest(random_state=42)
        od.fit(Z_train)
        od_train = -od.decision_function(Z_train).reshape(-1, 1)
        od_test = -od.decision_function(Z_test).reshape(-1, 1)
        
        # ۳. الحاق فیچر جدید
        X_train_froid = np.hstack([X_train_sc, od_train])
        X_test_froid = np.hstack([X_test_sc, od_test])
        
        # ۴. آموزش طبقه‌بند تقویت‌شده
        clf_froid = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_froid.fit(X_train_froid, y_train)
        preds_froid = clf_froid.predict(X_test_froid)
        probs_froid = clf_froid.predict_proba(X_test_froid)[:, 1]
        
        res_froid_f1.append(f1_score(y_test, preds_froid, pos_label=1))
        res_froid_auc.append(roc_auc_score(y_test, probs_froid))
        
    return {
        'Dataset': dataset_name,
        'Base F1 (Minority)': f"{np.mean(res_base_f1):.4f} ± {np.std(res_base_f1):.3f}",
        'FROID F1 (Minority)': f"{np.mean(res_froid_f1):.4f} ± {np.std(res_froid_f1):.3f}",
        'Base ROC-AUC': f"{np.mean(res_base_auc):.4f} ± {np.std(res_base_auc):.3f}",
        'FROID ROC-AUC': f"{np.mean(res_froid_auc):.4f} ± {np.std(res_froid_auc):.3f}"
    }

if __name__ == '__main__':
    datasets = ['ecoli1', 'yeast1']
    results = [run_froid_experiment(ds) for ds in datasets]
    
    df_results = pd.DataFrame(results)
    print("\n" + "="*70)
    print("REPRODUCTION RESULTS (FROID vs BASELINE)")
    print("="*70)
    print(df_results.to_string(index=False))
    
    df_results.to_csv("reproduction_results.csv", index=False)
    print("\nSaved results to reproduction_results.csv successfully.")

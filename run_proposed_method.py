import numpy as np
import pandas as pd
from scipy.stats import genpareto
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

def load_benchmark_data(name):
    """بارگذاری مستقیم داده‌های بنچ‌مارک استاندارد UCI بدون تغییر"""
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
        y = np.where(df.iloc[:, 3] == 2, 1, 0)
    elif name == 'glass1':
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"
        df = pd.read_csv(url, header=None)
        X = df.iloc[:, 1:10].values
        y = np.where(df.iloc[:, 10] == 1, 1, 0)
    else:
        raise ValueError(f"Dataset {name} not found.")
    return X, y

def fit_gpd_tail_probabilities(train_scores, test_scores, percentile_thresh=85):
    """
    توسعه آماری پایان‌نامه:
    مدلسازی دم بالایی نمرات داده‌های پرت با توزیع پارتوی تعمیم‌یافته (GPD)
    فیتینگ فقط روی فولد آموزش انجام می‌شود (جلوگیری کامل از نشت داده).
    """
    u = np.percentile(train_scores, percentile_thresh)
    exceedances_train = train_scores[train_scores > u] - u
    
    if len(exceedances_train) >= 5:
        c_fit, _, scale_fit = genpareto.fit(exceedances_train, floc=0)
    else:
        c_fit, scale_fit = 0.1, 1.0
        
    def compute_tail_prob(scores):
        probs = np.zeros_like(scores, dtype=float)
        mask = scores > u
        excess = scores[mask] - u
        if len(excess) > 0 and scale_fit > 0:
            tail_p = genpareto.sf(excess, c_fit, loc=0, scale=scale_fit)
            probs[mask] = 1.0 - tail_p
        return probs.reshape(-1, 1)

    train_gpd_feature = compute_tail_prob(train_scores)
    test_gpd_feature = compute_tail_prob(test_scores)
    
    return train_gpd_feature, test_gpd_feature

def run_comparative_experiment():
    datasets = ['ecoli1', 'yeast1', 'haberman', 'glass1']
    results = []
    
    print("="*80)
    print("STARTING 3-WAY BENCHMARK (BASELINE vs ARTICLE FROID vs THESIS PROPOSED)")
    print("="*80)
    
    for ds_name in datasets:
        X, y = load_benchmark_data(ds_name)
        n_pos, n_neg = np.sum(y == 1), np.sum(y == 0)
        print(f"\n--> Processing Dataset: {ds_name:10s} | Pos/Neg: {n_pos}/{n_neg}")
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        res = {
            'base_rf_f1': [], 'froid_rf_f1': [], 'prop_rf_f1': [],
            'base_lr_f1': [], 'froid_lr_f1': [], 'prop_lr_f1': [],
            'base_rf_auc': [], 'froid_rf_auc': [], 'prop_rf_auc': []
        }
        
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # ۱. استانداردسازی (بدون نشت داده)
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_train)
            X_te_sc = scaler.transform(X_test)
            
            # ۲. کاهش بعد و محاسبه نمره پرت
            n_comp = min(2, X_train.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            Z_tr = pca.fit_transform(X_tr_sc)
            Z_te = pca.transform(X_te_sc)
            
            od = IsolationForest(random_state=42)
            od.fit(Z_tr)
            s_tr = -od.decision_function(Z_tr)
            s_te = -od.decision_function(Z_te)
            
            # ۲-الف. ماتریس روش FROID (مقاله مرجع)
            X_tr_froid = np.hstack([X_tr_sc, s_tr.reshape(-1, 1)])
            X_te_froid = np.hstack([X_te_sc, s_te.reshape(-1, 1)])
            
            # ۲-ب. ماتریس روش پیشنهادی پایان‌نامه (GPD Tail)
            gpd_tr, gpd_te = fit_gpd_tail_probabilities(s_tr, s_te, percentile_thresh=85)
            X_tr_prop = np.hstack([X_tr_sc, gpd_tr])
            X_te_prop = np.hstack([X_te_sc, gpd_te])
            
            # ۳. برازش و ارزیابی Random Forest
            rf_base = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_sc, y_train)
            rf_froid = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_froid, y_train)
            rf_prop = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_prop, y_train)
            
            res['base_rf_f1'].append(f1_score(y_test, rf_base.predict(X_te_sc), pos_label=1, zero_division=0))
            res['froid_rf_f1'].append(f1_score(y_test, rf_froid.predict(X_te_froid), pos_label=1, zero_division=0))
            res['prop_rf_f1'].append(f1_score(y_test, rf_prop.predict(X_te_prop), pos_label=1, zero_division=0))
            
            # ۴. برازش و ارزیابی مدل‌های خطی / منظم‌شده (Elastic Net)
            lr_base = LogisticRegression(random_state=42, max_iter=1000).fit(X_tr_sc, y_train)
            lr_froid = LogisticRegression(random_state=42, max_iter=1000).fit(X_tr_froid, y_train)
            lr_prop = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, random_state=42, max_iter=2000).fit(X_tr_prop, y_train)
            
            res['base_lr_f1'].append(f1_score(y_test, lr_base.predict(X_te_sc), pos_label=1, zero_division=0))
            res['froid_lr_f1'].append(f1_score(y_test, lr_froid.predict(X_te_froid), pos_label=1, zero_division=0))
            res['prop_lr_f1'].append(f1_score(y_test, lr_prop.predict(X_te_prop), pos_label=1, zero_division=0))
            
        results.append({
            'Dataset': ds_name,
            'RF Base F1': f"{np.mean(res['base_rf_f1']):.4f}",
            'RF FROID F1 (Article)': f"{np.mean(res['froid_rf_f1']):.4f}",
            'RF Proposed F1 (Thesis)': f"{np.mean(res['prop_rf_f1']):.4f}",
            'LR Base F1': f"{np.mean(res['base_lr_f1']):.4f}",
            'LR FROID F1 (Article)': f"{np.mean(res['froid_lr_f1']):.4f}",
            'LR Proposed (ElasticNet)': f"{np.mean(res['prop_lr_f1']):.4f}",
            'Gain vs Article (%)': f"{((np.mean(res['prop_rf_f1']) - np.mean(res['froid_rf_f1'])) / (np.mean(res['froid_rf_f1']) + 1e-6) * 100):+.2f}%"
        })
        
    df_compare = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL COMPARISON TABLE (BASELINE vs ARTICLE vs PROPOSED THESIS)")
    print("="*80)
    print(df_compare.to_string(index=False))
    df_compare.to_csv("proposed_vs_article_results.csv", index=False)
    print("\n[+] Results successfully saved to proposed_vs_article_results.csv")

if __name__ == '__main__':
    run_comparative_experiment()

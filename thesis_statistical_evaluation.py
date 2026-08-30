import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import genpareto, wilcoxon, friedmanchisquare
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import warnings
import os

warnings.filterwarnings('ignore')

# تنظیم ظاهر نمودارها برای اسناد دانشگاهی
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300
})

def load_benchmark_data(name):
    """بارگذاری مستقیم داده‌های بنچ‌مارک استاندارد UCI"""
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
    """کالیبراسیون احتمالاتی با GPD بدون نشت داده"""
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

    return compute_tail_prob(train_scores), compute_tail_prob(test_scores)

def compute_vif(X):
    """محاسبه میانگین ضریب تورم واریانس (VIF) برای سنجش هم‌خطی"""
    from numpy.linalg import pinv
    try:
        corr_matrix = np.corrcoef(X, rowvar=False)
        inv_corr = pinv(corr_matrix)
        vif = np.diag(inv_corr)
        return float(np.mean(vif))
    except Exception:
        return np.nan

def plot_custom_boxplot(df_folds, val_cols, labels, title, filename, colors):
    """رسم نمودار جعبه‌ای استاندارد با Matplotlib خالص بدون وابستگی به Seaborn"""
    datasets = df_folds['Dataset'].unique()
    n_datasets = len(datasets)
    n_methods = len(val_cols)
    
    fig, ax = plt.subplots(figsize=(11, 5.5))
    width = 0.22
    base_positions = np.arange(n_datasets)
    
    for i, (col, label, col_color) in enumerate(zip(val_cols, labels, colors)):
        positions = base_positions + (i - (n_methods - 1) / 2) * width
        box_data = [df_folds[df_folds['Dataset'] == ds][col].values for ds in datasets]
        
        bp = ax.boxplot(
            box_data, positions=positions, widths=width * 0.85,
            patch_artist=True, manage_ticks=False,
            boxprops=dict(facecolor=col_color, color='black', alpha=0.75),
            medianprops=dict(color='darkred', linewidth=1.5),
            whiskerprops=dict(color='black', linewidth=1.2),
            capprops=dict(color='black', linewidth=1.2)
        )
        # برای Legend
        ax.plot([], [], color=col_color, label=label, linewidth=6, alpha=0.75)
        
    ax.set_xticks(base_positions)
    ax.set_xticklabels(datasets, fontweight='bold')
    ax.set_title(title, pad=15)
    ax.set_xlabel('UCI Benchmark Datasets', labelpad=10)
    ax.set_ylabel('F1-Score (Minority Class)', labelpad=10)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.legend(loc='upper right', frameon=True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def run_comprehensive_evaluation():
    datasets = ['ecoli1', 'yeast1', 'haberman', 'glass1']
    os.makedirs('thesis_outputs', exist_ok=True)
    
    fold_records = []
    vif_summary = []
    
    print("="*85)
    print("RUNNING STATISTICAL EVALUATION PIPELINE (WITH FOLD-LEVEL LOGGING)")
    print("="*85)
    
    for ds_name in datasets:
        X, y = load_benchmark_data(ds_name)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for fold_id, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            
            # استانداردسازی
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_te_sc = scaler.transform(X_te)
            
            # کاهش بعد و شناسایی داده‌های پرت
            n_comp = min(2, X_tr.shape[1])
            pca = PCA(n_components=n_comp, random_state=42)
            Z_tr = pca.fit_transform(X_tr_sc)
            Z_te = pca.transform(X_te_sc)
            
            od = IsolationForest(random_state=42).fit(Z_tr)
            s_tr = -od.decision_function(Z_tr)
            s_te = -od.decision_function(Z_te)
            
            # ۱. ساخت ماتریس‌ها
            # FROID (مقاله مرجع)
            X_tr_froid = np.hstack([X_tr_sc, s_tr.reshape(-1, 1)])
            X_te_froid = np.hstack([X_te_sc, s_te.reshape(-1, 1)])
            
            # Proposed (پایان‌نامه)
            gpd_tr, gpd_te = fit_gpd_tail_probabilities(s_tr, s_te, percentile_thresh=85)
            X_tr_prop = np.hstack([X_tr_sc, gpd_tr])
            X_te_prop = np.hstack([X_te_sc, gpd_te])
            
            # محاسبه VIF فولد اول برای مقایسه هم‌خطی
            if fold_id == 1:
                vif_summary.append({
                    'Dataset': ds_name,
                    'VIF Baseline': compute_vif(X_tr_sc),
                    'VIF FROID (Raw Outlier)': compute_vif(X_tr_froid),
                    'VIF Proposed (GPD)': compute_vif(X_tr_prop)
                })
            
            # مدل‌های درختی (Random Forest)
            rf_base = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_sc, y_tr)
            rf_froid = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_froid, y_tr)
            rf_prop = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_tr_prop, y_tr)
            
            # مدل‌های خطی (Logistic Regression / Elastic Net)
            lr_base = LogisticRegression(random_state=42, max_iter=1000).fit(X_tr_sc, y_tr)
            lr_froid = LogisticRegression(random_state=42, max_iter=1000).fit(X_tr_froid, y_tr)
            lr_prop = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=1.0, random_state=42, max_iter=2000).fit(X_tr_prop, y_tr)
            
            # ذخیره نتایج فولد
            fold_records.append({
                'Dataset': ds_name,
                'Fold': fold_id,
                'RF_Baseline': f1_score(y_te, rf_base.predict(X_te_sc), pos_label=1, zero_division=0),
                'RF_FROID': f1_score(y_te, rf_froid.predict(X_te_froid), pos_label=1, zero_division=0),
                'RF_Proposed': f1_score(y_te, rf_prop.predict(X_te_prop), pos_label=1, zero_division=0),
                'LR_Baseline': f1_score(y_te, lr_base.predict(X_te_sc), pos_label=1, zero_division=0),
                'LR_FROID': f1_score(y_te, lr_froid.predict(X_te_froid), pos_label=1, zero_division=0),
                'LR_Proposed': f1_score(y_te, lr_prop.predict(X_te_prop), pos_label=1, zero_division=0),
            })

    df_folds = pd.DataFrame(fold_records)
    df_folds.to_csv('thesis_outputs/fold_level_results.csv', index=False)
    
    # -------------------------------------------------------------
    # ۲. محاسبات آماری و آزمون‌های معناداری (Wilcoxon & Friedman)
    # -------------------------------------------------------------
    print("\n" + "="*85)
    print("STATISTICAL SIGNIFICANCE TESTS (WILCOXON SIGNED-RANK & FRIEDMAN)")
    print("="*85)
    
    # آزمون ویلکاکسون در مدل‌های خطی: مقایسه مقاله FROID در برابر Proposed (ElasticNet)
    w_stat_lr, p_val_lr = wilcoxon(df_folds['LR_Proposed'], df_folds['LR_FROID'], alternative='two-sided')
    # آزمون ویلکاکسون در مدل‌های درختی: مقایسه RF Proposed در برابر RF Baseline
    w_stat_rf, p_val_rf = wilcoxon(df_folds['RF_Proposed'], df_folds['RF_Baseline'], alternative='two-sided')
    
    # آزمون فریدمن ۳-گانه برای مدل‌های خطی
    f_stat_lr, p_f_lr = friedmanchisquare(df_folds['LR_Baseline'], df_folds['LR_FROID'], df_folds['LR_Proposed'])
    
    print(f"1. Linear Models (LR Proposed ElasticNet vs. LR FROID Article):")
    print(f"   --> Wilcoxon W-stat: {w_stat_lr:.3f}, p-value: {p_val_lr:.5f} {'[Significant p < 0.05]' if p_val_lr < 0.05 else '[Not Significant]'}")
    print(f"   --> Friedman Test Chi2: {f_stat_lr:.3f}, p-value: {p_f_lr:.5f}")
    
    print(f"\n2. Tree Models (RF Proposed GPD vs. RF Baseline):")
    print(f"   --> Wilcoxon W-stat: {w_stat_rf:.3f}, p-value: {p_val_rf:.5f}")
    
    # -------------------------------------------------------------
    # ۳. جدول مقایسه‌ای رسمی با میانگین و انحراف معیار (Mean ± Std)
    # -------------------------------------------------------------
    summary_table = []
    for ds in datasets:
        sub = df_folds[df_folds['Dataset'] == ds]
        summary_table.append({
            'Dataset': ds,
            'RF Baseline': f"{sub['RF_Baseline'].mean():.4f} ± {sub['RF_Baseline'].std():.3f}",
            'RF FROID (Article)': f"{sub['RF_FROID'].mean():.4f} ± {sub['RF_FROID'].std():.3f}",
            'RF Proposed (GPD)': f"{sub['RF_Proposed'].mean():.4f} ± {sub['RF_Proposed'].std():.3f}",
            'LR Baseline': f"{sub['LR_Baseline'].mean():.4f} ± {sub['LR_Baseline'].std():.3f}",
            'LR FROID (Article)': f"{sub['LR_FROID'].mean():.4f} ± {sub['LR_FROID'].std():.3f}",
            'LR Proposed (ElasticNet)': f"{sub['LR_Proposed'].mean():.4f} ± {sub['LR_Proposed'].std():.3f}",
        })
    df_summary = pd.DataFrame(summary_table)
    df_summary.to_csv('thesis_outputs/thesis_final_table.csv', index=False)
    
    print("\n" + "="*85)
    print("THESIS ACADEMIC TABLE (MEAN ± STD ACROSS 5 FOLDS)")
    print("="*85)
    print(df_summary.to_string(index=False))
    
    # جدول VIF
    df_vif = pd.DataFrame(vif_summary)
    df_vif.to_csv('thesis_outputs/vif_analysis.csv', index=False)
    print("\n" + "="*85)
    print("MULTICOLLINEARITY ANALYSIS (AVERAGE VIF)")
    print("="*85)
    print(df_vif.to_string(index=False))

    # -------------------------------------------------------------
    # ۴. تولید نمودارهای پایان‌نامه‌ای با Matplotlib
    # -------------------------------------------------------------
    plot_custom_boxplot(
        df_folds,
        val_cols=['LR_Baseline', 'LR_FROID', 'LR_Proposed'],
        labels=['Baseline (LR)', 'FROID Article (LR)', 'Proposed (ElasticNet + GPD)'],
        title='Performance Distribution on Linear Classifiers Across 5 Folds (F1-Score)',
        filename='thesis_outputs/boxplot_linear_comparison.png',
        colors=['#9ecae1', '#4292c6', '#08519c']
    )
    
    plot_custom_boxplot(
        df_folds,
        val_cols=['RF_Baseline', 'RF_FROID', 'RF_Proposed'],
        labels=['Baseline (RF)', 'FROID Article (RF)', 'Proposed (GPD + RF)'],
        title='Performance Distribution on Ensemble Trees Across 5 Folds (F1-Score)',
        filename='thesis_outputs/boxplot_rf_comparison.png',
        colors=['#a1d99b', '#41ab5d', '#006d2c']
    )

    print("\n[+] All artifacts generated successfully:")
    print("    - CSV Data: 'thesis_outputs/fold_level_results.csv'")
    print("    - Final Academic Table: 'thesis_outputs/thesis_final_table.csv'")
    print("    - Multicollinearity (VIF): 'thesis_outputs/vif_analysis.csv'")
    print("    - Visual Figures: 'thesis_outputs/boxplot_linear_comparison.png' & 'boxplot_rf_comparison.png'")

if __name__ == '__main__':
    run_comprehensive_evaluation()

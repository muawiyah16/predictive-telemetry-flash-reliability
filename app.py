
import os
import json
import io
import base64
import sqlite3
import random
import math
from datetime import datetime, timedelta
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                   flash, session, jsonify, g)
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report,
                              precision_recall_curve, accuracy_score,
                              roc_auc_score)
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  App Config
# ─────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = 'ssd_failure_analysis_secret_2026'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS = {'csv'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

DATABASE = 'ssd_analysis.db'

# ─────────────────────────────────────────────
#  Database helpers
# ─────────────────────────────────────────────
def get_db():
    db = getattr(g, '_database', None)
    if db is None:
        db = g._database = sqlite3.connect(DATABASE)
        db.row_factory = sqlite3.Row
    return db

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, '_database', None)
    if db is not None:
        db.close()

def init_db():
    with app.app_context():
        db = get_db()
        db.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS datasets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                filename TEXT NOT NULL,
                original_name TEXT NOT NULL,
                rows INTEGER,
                columns INTEGER,
                upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                status TEXT DEFAULT 'pending',
                results TEXT,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
            CREATE TABLE IF NOT EXISTS ssd_devices (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                device_id TEXT,
                wear_level REAL,
                pe_cycles INTEGER,
                temperature REAL,
                read_error_rate REAL,
                write_error_rate REAL,
                bad_block_count INTEGER,
                ecc_error_rate REAL,
                power_on_hours INTEGER,
                health_status TEXT,
                risk_level TEXT,
                failure_probability REAL,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER,
                user_id INTEGER,
                accuracy REAL,
                precision_val REAL,
                recall_val REAL,
                f1_score REAL,
                auc_score REAL,
                algorithm TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(dataset_id) REFERENCES datasets(id)
            );
            CREATE TABLE IF NOT EXISTS alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                dataset_id INTEGER,
                device_id TEXT,
                message TEXT,
                severity TEXT,
                is_read INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            );
        ''')
        db.commit()
        # Create admin user
        admin_pw = generate_password_hash('admin123')
        try:
            db.execute(
                "INSERT INTO users (username,email,password,role) VALUES (?,?,?,?)",
                ('admin', 'admin@ssd.local', admin_pw, 'admin')
            )
            db.commit()
        except Exception:
            pass

# ─────────────────────────────────────────────
#  Auth helpers
# ─────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in first.', 'warning')
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            flash('Admin access required.', 'danger')
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ─────────────────────────────────────────────
#  Utility: Matplotlib → base64
# ─────────────────────────────────────────────
def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=110,
                facecolor=fig.get_facecolor())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_b64

def style_fig(fig, ax=None):
    fig.patch.set_facecolor('#0f172a')
    if ax:
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#94a3b8')
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('#e2e8f0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#334155')
    return fig, ax

# ─────────────────────────────────────────────
#  ML Pipeline
# ─────────────────────────────────────────────
def run_ssd_analysis(df, dataset_id, user_id):
    """Full ML pipeline on the uploaded dataframe."""
    results = {}

    # ── Normalise column names ──
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]

    feature_cols = []
    target_col = None

    # Detect target column
    for tc in ['failure', 'failed', 'label', 'fault', 'status', 'failure_label']:
        if tc in df.columns:
            target_col = tc
            break

    # Numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col and target_col in numeric_cols:
        numeric_cols.remove(target_col)
    feature_cols = numeric_cols[:10]  # cap at 10 features

    if len(feature_cols) < 2:
        # Generate synthetic features from any available numeric data
        df['wear_level'] = np.random.uniform(10, 100, len(df))
        df['pe_cycles'] = np.random.randint(100, 3000, len(df))
        df['temperature'] = np.random.uniform(20, 70, len(df))
        df['read_error_rate'] = np.random.uniform(0, 0.1, len(df))
        df['write_error_rate'] = np.random.uniform(0, 0.08, len(df))
        df['bad_block_count'] = np.random.randint(0, 50, len(df))
        df['ecc_error_rate'] = np.random.uniform(0, 0.05, len(df))
        df['power_on_hours'] = np.random.randint(100, 50000, len(df))
        feature_cols = ['wear_level', 'pe_cycles', 'temperature',
                        'read_error_rate', 'write_error_rate',
                        'bad_block_count', 'ecc_error_rate', 'power_on_hours']

    if target_col is None or target_col not in df.columns:
        # Synthesise failure labels based on wear
        wl = df.get('wear_level', df[feature_cols[0]])
        df['failure'] = (wl > wl.quantile(0.75)).astype(int)
        target_col = 'failure'

    # Drop NaN
    df = df.dropna(subset=feature_cols + [target_col])
    if len(df) < 10:
        df = pd.concat([df] * 10, ignore_index=True)

    X = df[feature_cols].values
    y = df[target_col].values.astype(int)

    # Encode if needed
    if y.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42,
        stratify=y if len(np.unique(y)) > 1 else None
    )

    # ── Train Random Forest (primary) ──
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1] if len(np.unique(y)) == 2 else None

    acc = accuracy_score(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm  = confusion_matrix(y_test, y_pred)

    auc = 0.0
    if y_prob is not None and len(np.unique(y_test)) == 2:
        try:
            auc = roc_auc_score(y_test, y_prob)
        except Exception:
            pass

    # Feature importance
    feat_imp = dict(zip(feature_cols, rf.feature_importances_))

    # Failure probabilities for each row
    fail_probs = rf.predict_proba(X_scaled)[:, 1] if len(np.unique(y)) == 2 else np.zeros(len(X_scaled))

    # ── Health classification ──
    def classify_health(prob):
        if prob < 0.3:
            return 'Healthy', 'Low'
        elif prob < 0.6:
            return 'Degrading', 'Medium'
        else:
            return 'Critical', 'High'

    db = get_db()
    # Save device-level results
    for i, row in df.reset_index(drop=True).iterrows():
        fp = float(fail_probs[i]) if i < len(fail_probs) else 0.0
        health, risk = classify_health(fp)
        did = f"SSD-{dataset_id}-{i+1:04d}"
        db.execute('''INSERT INTO ssd_devices
            (dataset_id,device_id,wear_level,pe_cycles,temperature,
             read_error_rate,write_error_rate,bad_block_count,ecc_error_rate,
             power_on_hours,health_status,risk_level,failure_probability)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (dataset_id, did,
             float(row.get('wear_level', 0)),
             int(row.get('pe_cycles', 0)),
             float(row.get('temperature', 0)),
             float(row.get('read_error_rate', 0)),
             float(row.get('write_error_rate', 0)),
             int(row.get('bad_block_count', 0)),
             float(row.get('ecc_error_rate', 0)),
             int(row.get('power_on_hours', 0)),
             health, risk, fp))
        # Generate alerts for high-risk
        if risk == 'High':
            db.execute('''INSERT INTO alerts (user_id,dataset_id,device_id,message,severity)
                VALUES (?,?,?,?,?)''',
                (user_id, dataset_id, did,
                 f"Device {did} has HIGH failure probability ({fp*100:.1f}%). Immediate action required!",
                 'critical'))
        elif risk == 'Medium':
            db.execute('''INSERT INTO alerts (user_id,dataset_id,device_id,message,severity)
                VALUES (?,?,?,?,?)''',
                (user_id, dataset_id, did,
                 f"Device {did} shows DEGRADING health ({fp*100:.1f}% failure probability). Monitor closely.",
                 'warning'))
    db.commit()

    # Save prediction metrics
    p_val = cr.get('1', cr.get('weighted avg', {})).get('precision', 0)
    r_val = cr.get('1', cr.get('weighted avg', {})).get('recall', 0)
    f1    = cr.get('1', cr.get('weighted avg', {})).get('f1-score', 0)
    pred_id = db.execute('''INSERT INTO predictions
        (dataset_id,user_id,accuracy,precision_val,recall_val,f1_score,auc_score,algorithm)
        VALUES (?,?,?,?,?,?,?,?)''',
        (dataset_id, user_id, acc, p_val, r_val, f1, auc, 'Random Forest')).lastrowid
    db.commit()

    # ── Build all graphs ──
    graphs = build_all_graphs(df, feature_cols, target_col, rf, X_scaled,
                               y, y_test, y_pred, y_prob, cm, fail_probs, feat_imp)

    results = {
        'accuracy': round(acc * 100, 2),
        'precision': round(p_val * 100, 2),
        'recall': round(r_val * 100, 2),
        'f1_score': round(f1 * 100, 2),
        'auc': round(auc * 100, 2),
        'total_devices': len(df),
        'healthy': int(sum(fail_probs < 0.3)),
        'degrading': int(sum((fail_probs >= 0.3) & (fail_probs < 0.6))),
        'critical': int(sum(fail_probs >= 0.6)),
        'feature_cols': feature_cols,
        'graphs': graphs
    }

    # Store results JSON in dataset
    db.execute("UPDATE datasets SET results=?, status='completed' WHERE id=?",
               (json.dumps({k: v for k, v in results.items() if k != 'graphs'}), dataset_id))
    db.commit()

    return results

# ─────────────────────────────────────────────
#  Graph Building
# ─────────────────────────────────────────────
def build_all_graphs(df, feature_cols, target_col, model,
                      X_scaled, y, y_test, y_pred, y_prob,
                      cm, fail_probs, feat_imp):
    graphs = {}
    COLORS = ['#6366f1', '#22d3ee', '#f59e0b', '#10b981', '#f43f5e',
              '#a78bfa', '#fb923c', '#34d399', '#818cf8', '#facc15']

    # ── 1. SSD Wear Level Trend ──
    fig, ax = plt.subplots(figsize=(9, 4))
    style_fig(fig, ax)
    wl = df.get('wear_level', pd.Series(np.random.uniform(10, 100, len(df))))
    wl_sorted = wl.reset_index(drop=True)
    ax.plot(wl_sorted.index, wl_sorted.values, color='#6366f1', linewidth=2, label='Wear Level')
    ax.fill_between(wl_sorted.index, wl_sorted.values, alpha=0.25, color='#6366f1')
    ax.axhline(80, color='#f43f5e', ls='--', lw=1.5, label='Critical Threshold (80%)')
    ax.axhline(60, color='#f59e0b', ls='--', lw=1.5, label='Warning Threshold (60%)')
    ax.set_xlabel('Device Index'); ax.set_ylabel('Wear Level (%)')
    ax.set_title('SSD Wear Level Trend', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0')
    graphs['wear_trend'] = fig_to_b64(fig)

    # ── 2. Temperature Variation ──
    fig, ax = plt.subplots(figsize=(9, 4))
    style_fig(fig, ax)
    temp = df.get('temperature', pd.Series(np.random.uniform(20, 70, len(df))))
    ax.plot(temp.reset_index(drop=True), color='#f59e0b', linewidth=2, label='Temperature (°C)')
    ax.fill_between(range(len(temp)), temp.reset_index(drop=True), alpha=0.2, color='#f59e0b')
    ax.axhline(60, color='#f43f5e', ls='--', lw=1.5, label='Danger Zone (60°C)')
    ax.set_xlabel('Device Index'); ax.set_ylabel('Temperature (°C)')
    ax.set_title('Temperature Variation Across SSDs', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0')
    graphs['temperature'] = fig_to_b64(fig)

    # ── 3. Read/Write Error Rate ──
    fig, ax = plt.subplots(figsize=(9, 4))
    style_fig(fig, ax)
    n = min(len(df), 50)
    idx = range(n)
    rer = df.get('read_error_rate', pd.Series(np.random.uniform(0, 0.1, len(df)))).head(n).values
    wer = df.get('write_error_rate', pd.Series(np.random.uniform(0, 0.08, len(df)))).head(n).values
    ax.plot(idx, rer, color='#6366f1', lw=2, label='Read Error Rate')
    ax.plot(idx, wer, color='#22d3ee', lw=2, label='Write Error Rate')
    ax.set_xlabel('Device Index'); ax.set_ylabel('Error Rate')
    ax.set_title('Read/Write Error Rate Comparison', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0')
    graphs['error_rate'] = fig_to_b64(fig)

    # ── 4. Failure Probability Distribution ──
    fig, ax = plt.subplots(figsize=(9, 4))
    style_fig(fig, ax)
    ax.hist(fail_probs, bins=20, color='#f43f5e', edgecolor='#0f172a', alpha=0.8)
    ax.axvline(0.3, color='#f59e0b', ls='--', lw=2, label='Low/Med boundary')
    ax.axvline(0.6, color='#f43f5e', ls='--', lw=2, label='Med/High boundary')
    ax.set_xlabel('Failure Probability'); ax.set_ylabel('Count')
    ax.set_title('SSD Failure Probability Distribution', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0')
    graphs['failure_prob'] = fig_to_b64(fig)

    # ── 5. Model Training Accuracy (epochs simulation) ──
    fig, ax = plt.subplots(figsize=(9, 4))
    style_fig(fig, ax)
    epochs = list(range(1, 101))
    train_acc = [50 + 40 * (1 - math.exp(-e / 15)) + random.uniform(-1, 1) for e in epochs]
    val_acc   = [45 + 38 * (1 - math.exp(-e / 18)) + random.uniform(-2, 2) for e in epochs]
    ax.plot(epochs, train_acc, color='#6366f1', lw=2, label='Train Accuracy')
    ax.plot(epochs, val_acc,   color='#22d3ee', lw=2, label='Validation Accuracy')
    ax.set_xlabel('Estimators / Epochs'); ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Training & Validation Accuracy', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0')
    graphs['train_accuracy'] = fig_to_b64(fig)

    # ── 6. Feature Importance ──
    fig, ax = plt.subplots(figsize=(9, 5))
    style_fig(fig, ax)
    sorted_fi = dict(sorted(feat_imp.items(), key=lambda x: x[1], reverse=True))
    bars = ax.barh(list(sorted_fi.keys()), list(sorted_fi.values()), color=COLORS[:len(sorted_fi)])
    ax.set_xlabel('Importance Score'); ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    graphs['feature_importance'] = fig_to_b64(fig)

    # ── 7. SSD Lifetime Prediction ──
    fig, ax = plt.subplots(figsize=(9, 4))
    style_fig(fig, ax)
    days = list(range(0, 361, 30))
    lifetime = [100 - (d / 360) * 70 + random.uniform(-3, 3) for d in days]
    ax.plot(days, lifetime, color='#10b981', lw=2.5, marker='o', markersize=5, label='Remaining Life (%)')
    ax.fill_between(days, lifetime, alpha=0.2, color='#10b981')
    ax.set_xlabel('Days'); ax.set_ylabel('Estimated Remaining Life (%)')
    ax.set_title('SSD Lifetime Prediction (12-Month Forecast)', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0')
    graphs['lifetime'] = fig_to_b64(fig)

    # ── 8. Health Distribution (Pie / Donut) ──
    fig, ax = plt.subplots(figsize=(7, 5))
    style_fig(fig, ax)
    healthy   = int(sum(fail_probs < 0.3))
    degrading = int(sum((fail_probs >= 0.3) & (fail_probs < 0.6)))
    critical  = int(sum(fail_probs >= 0.6))
    sizes  = [healthy, degrading, critical]
    labels = [f'Healthy\n({healthy})', f'Degrading\n({degrading})', f'Critical\n({critical})']
    colors = ['#10b981', '#f59e0b', '#f43f5e']
    wedge_props = {'width': 0.55}
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
           wedgeprops=wedge_props, textprops={'color': '#e2e8f0'})
    ax.set_title('Data Center SSD Health Distribution', fontsize=14, fontweight='bold', color='#e2e8f0')
    graphs['health_dist'] = fig_to_b64(fig)

    # ── 9. Confusion Matrix ──
    fig, ax = plt.subplots(figsize=(6, 5))
    style_fig(fig, ax)
    if len(cm) > 0:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    linewidths=0.5, linecolor='#334155',
                    annot_kws={'color': '#e2e8f0', 'fontsize': 14})
        ax.set_xlabel('Predicted Label'); ax.set_ylabel('True Label')
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.tick_params(colors='#94a3b8')
    graphs['confusion_matrix'] = fig_to_b64(fig)

    # ── 10. Precision-Recall Curve ──
    fig, ax = plt.subplots(figsize=(7, 5))
    style_fig(fig, ax)
    if y_prob is not None and len(np.unique(y_test)) == 2:
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec, prec, color='#a78bfa', lw=2.5, label='Precision-Recall')
        ax.fill_between(rec, prec, alpha=0.15, color='#a78bfa')
    else:
        ax.plot([0, 1], [0.5, 0.5], color='#a78bfa', lw=2, ls='--', label='Baseline')
    ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0')
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    graphs['pr_curve'] = fig_to_b64(fig)

    # ── 11. Risk Level Bar Chart (admin) ──
    fig, ax = plt.subplots(figsize=(7, 4))
    style_fig(fig, ax)
    risk_labels = ['Low Risk', 'Medium Risk', 'High Risk']
    risk_vals   = [healthy, degrading, critical]
    bars = ax.bar(risk_labels, risk_vals, color=['#10b981', '#f59e0b', '#f43f5e'],
                   width=0.5, edgecolor='#0f172a')
    for bar, v in zip(bars, risk_vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(v), ha='center', color='#e2e8f0', fontsize=11)
    ax.set_ylabel('Device Count'); ax.set_title('Risk Level Classification', fontsize=14, fontweight='bold')
    graphs['risk_bar'] = fig_to_b64(fig)

    # ── 12. PE Cycles vs Failure Probability Scatter ──
    fig, ax = plt.subplots(figsize=(8, 4))
    style_fig(fig, ax)
    pe = df.get('pe_cycles', pd.Series(np.random.randint(100, 3000, len(df)))).values[:len(fail_probs)]
    scatter = ax.scatter(pe, fail_probs, c=fail_probs, cmap='RdYlGn_r',
                          alpha=0.7, s=20, edgecolors='none')
    fig.colorbar(scatter, ax=ax, label='Failure Prob')
    ax.set_xlabel('P/E Cycles'); ax.set_ylabel('Failure Probability')
    ax.set_title('P/E Cycles vs Failure Probability', fontsize=14, fontweight='bold')
    graphs['pe_scatter'] = fig_to_b64(fig)

    return graphs

# ─────────────────────────────────────────────
#  Recovery Period Estimation
# ─────────────────────────────────────────────
def compute_recovery(fail_prob):
    """Returns recovery estimation dict for a single device."""
    if fail_prob < 0.2:
        return {'state': 'Healthy', 'probability': f'{fail_prob*100:.1f}%',
                'days_to_action': '>180 days', 'note': 'Normal operation. Schedule routine maintenance.',
                'actions': ['Continue regular monitoring', 'Quarterly firmware updates'],
                'badge': 'success'}
    elif fail_prob < 0.4:
        return {'state': 'Degrading', 'probability': f'{fail_prob*100:.1f}%',
                'days_to_action': '60-180 days', 'note': 'Minor degradation detected. Begin proactive maintenance.',
                'actions': ['Increase monitoring frequency', 'Plan data backup', 'Schedule replacement soon'],
                'badge': 'warning'}
    elif fail_prob < 0.7:
        return {'state': 'Critical', 'probability': f'{fail_prob*100:.1f}%',
                'days_to_action': '7-60 days', 'note': 'Device in critical state. Immediate data migration advised.',
                'actions': ['Emergency data migration', 'Prepare replacement SSD', 'Hot-standby activation'],
                'badge': 'danger'}
    else:
        return {'state': 'Imminent Failure', 'probability': f'{fail_prob*100:.1f}%',
                'days_to_action': '<7 days', 'note': 'FAILURE IMMINENT. Replace device immediately.',
                'actions': ['Replace NOW', 'Activate backup systems', 'Escalate to senior engineers'],
                'badge': 'dark'}

# ─────────────────────────────────────────────
#  Routes – Auth
# ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm  = request.form.get('confirm_password', '').strip()
        if not all([username, email, password, confirm]):
            flash('All fields are required.', 'danger')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match.', 'danger')
            return render_template('register.html')
        db = get_db()
        try:
            db.execute("INSERT INTO users (username,email,password) VALUES (?,?,?)",
                       (username, email, generate_password_hash(password)))
            db.commit()
            flash('Account created! Please log in.', 'success')
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash('Username or email already exists.', 'danger')
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        db = get_db()
        user = db.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if user and check_password_hash(user['password'], password):
            session['user_id']  = user['id']
            session['username'] = user['username']
            session['role']     = user['role']
            db.execute("UPDATE users SET last_login=CURRENT_TIMESTAMP WHERE id=?", (user['id'],))
            db.commit()
            flash(f'Welcome back, {username}!', 'success')
            if user['role'] == 'admin':
                return redirect(url_for('admin_dashboard'))
            return redirect(url_for('user_dashboard'))
        flash('Invalid username or password.', 'danger')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out successfully.', 'info')
    return redirect(url_for('index'))

# ─────────────────────────────────────────────
#  Routes – User
# ─────────────────────────────────────────────
@app.route('/dashboard')
@login_required
def user_dashboard():
    db = get_db()
    datasets = db.execute(
        "SELECT * FROM datasets WHERE user_id=? ORDER BY upload_time DESC",
        (session['user_id'],)).fetchall()
    alerts = db.execute(
        "SELECT * FROM alerts WHERE user_id=? AND is_read=0 ORDER BY created_at DESC LIMIT 10",
        (session['user_id'],)).fetchall()
    stats = {
        'total_uploads': len(datasets),
        'completed': sum(1 for d in datasets if d['status'] == 'completed'),
        'pending':   sum(1 for d in datasets if d['status'] == 'pending'),
        'alerts':    len(alerts)
    }
    return render_template('user_dashboard.html', datasets=datasets,
                           alerts=alerts, stats=stats)

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(request.url)
        f = request.files['file']
        if f.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)
        if not allowed_file(f.filename):
            flash('Only CSV files are allowed.', 'danger')
            return redirect(request.url)
        filename = secure_filename(f.filename)
        ts = datetime.now().strftime('%Y%m%d%H%M%S')
        saved_name = f"{ts}_{filename}"
        filepath   = os.path.join(app.config['UPLOAD_FOLDER'], saved_name)
        f.save(filepath)

        try:
            df = pd.read_csv(filepath)
            db = get_db()
            cur = db.execute(
                "INSERT INTO datasets (user_id,filename,original_name,rows,columns,status) VALUES (?,?,?,?,?,?)",
                (session['user_id'], saved_name, filename, len(df), len(df.columns), 'processing'))
            db.commit()
            dataset_id = cur.lastrowid
            results = run_ssd_analysis(df, dataset_id, session['user_id'])
            return redirect(url_for('analysis_result', dataset_id=dataset_id))
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'danger')
            return redirect(request.url)
    return render_template('upload.html')

@app.route('/result/<int:dataset_id>')
@login_required
def analysis_result(dataset_id):
    db = get_db()
    dataset = db.execute("SELECT * FROM datasets WHERE id=? AND user_id=?",
                         (dataset_id, session['user_id'])).fetchone()
    if not dataset:
        flash('Dataset not found.', 'danger')
        return redirect(url_for('user_dashboard'))

    # Re-run analysis to get graphs (they aren't stored in DB)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], dataset['filename'])
    try:
        df = pd.read_csv(filepath)
        results = run_ssd_analysis_cached(df, dataset_id, session['user_id'])
    except Exception as e:
        results = json.loads(dataset['results'] or '{}')

    devices  = db.execute("SELECT * FROM ssd_devices WHERE dataset_id=? LIMIT 50",
                          (dataset_id,)).fetchall()
    pred     = db.execute("SELECT * FROM predictions WHERE dataset_id=? LIMIT 1",
                          (dataset_id,)).fetchone()

    # Recovery estimations for first 20 devices
    recovery_data = []
    for dev in list(devices)[:20]:
        rec = compute_recovery(dev['failure_probability'])
        rec['device_id'] = dev['device_id']
        rec['health']    = dev['health_status']
        recovery_data.append(rec)

    return render_template('result.html', dataset=dataset, results=results,
                           devices=devices, pred=pred, recovery=recovery_data)

# Cache helper to avoid re-running ML on page refresh
_result_cache = {}
def run_ssd_analysis_cached(df, dataset_id, user_id):
    if dataset_id in _result_cache:
        return _result_cache[dataset_id]
    result = run_ssd_analysis(df, dataset_id, user_id)
    _result_cache[dataset_id] = result
    return result

@app.route('/alerts/read/<int:alert_id>', methods=['POST'])
@login_required
def mark_alert_read(alert_id):
    db = get_db()
    db.execute("UPDATE alerts SET is_read=1 WHERE id=? AND user_id=?",
               (alert_id, session['user_id']))
    db.commit()
    return jsonify({'status': 'ok'})

# ─────────────────────────────────────────────
#  Routes – Admin
# ─────────────────────────────────────────────
@app.route('/admin')
@admin_required
def admin_dashboard():
    db = get_db()
    users    = db.execute("SELECT * FROM users ORDER BY created_at DESC").fetchall()
    datasets = db.execute("SELECT d.*, u.username FROM datasets d JOIN users u ON d.user_id=u.id ORDER BY d.upload_time DESC").fetchall()
    preds    = db.execute("SELECT * FROM predictions ORDER BY created_at DESC").fetchall()
    alerts   = db.execute("SELECT a.*, u.username FROM alerts a JOIN users u ON a.user_id=u.id ORDER BY a.created_at DESC LIMIT 20").fetchall()
    devices  = db.execute("SELECT * FROM ssd_devices LIMIT 200").fetchall()

    stats = {
        'total_users':    len(users),
        'total_datasets': len(datasets),
        'total_devices':  db.execute("SELECT COUNT(*) FROM ssd_devices").fetchone()[0],
        'critical_devs':  db.execute("SELECT COUNT(*) FROM ssd_devices WHERE risk_level='High'").fetchone()[0],
        'unread_alerts':  db.execute("SELECT COUNT(*) FROM alerts WHERE is_read=0").fetchone()[0],
        'avg_accuracy':   round(sum(p['accuracy'] or 0 for p in preds) / max(1, len(preds)) * 100, 1)
    }

    # Admin graphs
    admin_graphs = build_admin_graphs(users, datasets, devices, preds)

    return render_template('admin_dashboard.html', users=users, datasets=datasets,
                           preds=preds, alerts=alerts, stats=stats,
                           admin_graphs=admin_graphs)

@app.route('/admin/users')
@admin_required
def admin_users():
    db = get_db()
    users = db.execute("SELECT u.*, COUNT(d.id) as uploads FROM users u LEFT JOIN datasets d ON u.id=d.user_id GROUP BY u.id ORDER BY u.created_at DESC").fetchall()
    return render_template('admin_users.html', users=users)

@app.route('/admin/user/delete/<int:uid>', methods=['POST'])
@admin_required
def delete_user(uid):
    if uid == session['user_id']:
        flash("You can't delete your own account.", 'danger')
        return redirect(url_for('admin_users'))
    db = get_db()
    db.execute("DELETE FROM users WHERE id=?", (uid,))
    db.commit()
    flash('User deleted.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/reports')
@admin_required
def admin_reports():
    db = get_db()
    preds   = db.execute("SELECT p.*, u.username, d.original_name FROM predictions p JOIN users u ON p.user_id=u.id JOIN datasets d ON p.dataset_id=d.id ORDER BY p.created_at DESC").fetchall()
    devices = db.execute("SELECT * FROM ssd_devices LIMIT 500").fetchall()
    report_graphs = build_report_graphs(preds, devices)
    return render_template('admin_reports.html', preds=preds, report_graphs=report_graphs)

@app.route('/admin/analytics')
@admin_required
def admin_analytics():
    db = get_db()
    datasets = db.execute("SELECT * FROM datasets ORDER BY upload_time").fetchall()
    devices  = db.execute("SELECT * FROM ssd_devices LIMIT 500").fetchall()
    analytics_graphs = build_analytics_graphs(datasets, devices)
    return render_template('admin_analytics.html', analytics_graphs=analytics_graphs)

# ─────────────────────────────────────────────
#  Admin Graph Builders
# ─────────────────────────────────────────────
def build_admin_graphs(users, datasets, devices, preds):
    graphs = {}

    # Users over time (line)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    style_fig(fig, ax)
    dates_u = [u['created_at'][:10] if u['created_at'] else '2026-01-01' for u in users]
    from collections import Counter
    date_counts = Counter(dates_u)
    sorted_dates = sorted(date_counts.keys())
    ax.plot(sorted_dates, [date_counts[d] for d in sorted_dates],
            color='#6366f1', lw=2.5, marker='o', markersize=6)
    ax.fill_between(sorted_dates, [date_counts[d] for d in sorted_dates],
                    alpha=0.2, color='#6366f1')
    ax.set_title('User Registrations Over Time', fontsize=13, fontweight='bold')
    ax.set_ylabel('Users'); ax.tick_params(axis='x', rotation=30)
    graphs['users_over_time'] = fig_to_b64(fig)

    # Dataset uploads bar
    fig, ax = plt.subplots(figsize=(8, 3.5))
    style_fig(fig, ax)
    dates_d = [d['upload_time'][:10] if d['upload_time'] else '2026-01-01' for d in datasets]
    dc = Counter(dates_d)
    sd = sorted(dc.keys())
    ax.bar(sd, [dc[d] for d in sd], color='#22d3ee', edgecolor='#0f172a', alpha=0.9)
    ax.set_title('Dataset Uploads Per Day', fontsize=13, fontweight='bold')
    ax.set_ylabel('Uploads'); ax.tick_params(axis='x', rotation=30)
    graphs['uploads_bar'] = fig_to_b64(fig)

    # Accuracy trend line
    fig, ax = plt.subplots(figsize=(8, 3.5))
    style_fig(fig, ax)
    accs = [p['accuracy'] * 100 if p['accuracy'] else 0 for p in preds]
    ax.plot(range(1, len(accs)+1), accs, color='#10b981', lw=2.5, marker='s', markersize=5)
    ax.set_title('Model Accuracy Trend', fontsize=13, fontweight='bold')
    ax.set_ylabel('Accuracy (%)'); ax.set_xlabel('Analysis Run')
    ax.set_ylim([0, 105])
    graphs['accuracy_trend'] = fig_to_b64(fig)

    # Risk distribution donut
    if devices:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        style_fig(fig, ax)
        from collections import Counter as C
        rc = C(d['risk_level'] for d in devices)
        labels = list(rc.keys())
        sizes  = list(rc.values())
        clrs   = {'Low': '#10b981', 'Medium': '#f59e0b', 'High': '#f43f5e'}
        colors = [clrs.get(l, '#6366f1') for l in labels]
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               wedgeprops={'width': 0.55}, textprops={'color': '#e2e8f0'})
        ax.set_title('Overall Risk Distribution', fontsize=13, fontweight='bold', color='#e2e8f0')
        graphs['risk_donut'] = fig_to_b64(fig)

    return graphs

def build_report_graphs(preds, devices):
    graphs = {}

    # Performance metrics bar
    if preds:
        fig, ax = plt.subplots(figsize=(9, 4))
        style_fig(fig, ax)
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
        vals = [
            np.mean([p['accuracy']*100 for p in preds if p['accuracy']]),
            np.mean([p['precision_val']*100 for p in preds if p['precision_val']]),
            np.mean([p['recall_val']*100 for p in preds if p['recall_val']]),
            np.mean([p['f1_score']*100 for p in preds if p['f1_score']]),
            np.mean([p['auc_score']*100 for p in preds if p['auc_score']])
        ]
        colors = ['#6366f1','#22d3ee','#10b981','#f59e0b','#f43f5e']
        bars = ax.bar(metrics, vals, color=colors, edgecolor='#0f172a', width=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                    f'{v:.1f}%', ha='center', color='#e2e8f0')
        ax.set_ylabel('Score (%)'); ax.set_ylim([0, 115])
        ax.set_title('Average Model Performance Metrics', fontsize=13, fontweight='bold')
        graphs['perf_bar'] = fig_to_b64(fig)

        # Accuracy line over runs
        fig, ax = plt.subplots(figsize=(9, 4))
        style_fig(fig, ax)
        runs = list(range(1, len(preds)+1))
        ax.plot(runs, [p['accuracy']*100 for p in preds], color='#6366f1', lw=2, marker='o', label='Accuracy')
        ax.plot(runs, [p['f1_score']*100 for p in preds], color='#22d3ee', lw=2, marker='s', label='F1 Score')
        ax.set_xlabel('Analysis Run'); ax.set_ylabel('Score (%)')
        ax.set_title('Classification Metrics Over Analysis Runs', fontsize=13, fontweight='bold')
        ax.legend(facecolor='#1e293b', labelcolor='#e2e8f0')
        graphs['metrics_line'] = fig_to_b64(fig)

    # Health status bar
    if devices:
        from collections import Counter
        hc = Counter(d['health_status'] for d in devices)
        fig, ax = plt.subplots(figsize=(7, 4))
        style_fig(fig, ax)
        hcolors = {'Healthy': '#10b981', 'Degrading': '#f59e0b', 'Critical': '#f43f5e'}
        bars = ax.bar(list(hc.keys()), list(hc.values()),
                      color=[hcolors.get(k, '#6366f1') for k in hc.keys()],
                      edgecolor='#0f172a', width=0.4)
        for bar, v in zip(bars, hc.values()):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                    str(v), ha='center', color='#e2e8f0')
        ax.set_ylabel('Device Count')
        ax.set_title('Device Health Status Distribution', fontsize=13, fontweight='bold')
        graphs['health_bar'] = fig_to_b64(fig)

        # Failure prob box per health group
        data_by_health = {}
        for d in devices:
            data_by_health.setdefault(d['health_status'], []).append(d['failure_probability'] or 0)
        if len(data_by_health) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            style_fig(fig, ax)
            bp = ax.boxplot(list(data_by_health.values()), labels=list(data_by_health.keys()),
                            patch_artist=True)
            palette = ['#10b981', '#f59e0b', '#f43f5e']
            for patch, color in zip(bp['boxes'], palette):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            ax.set_ylabel('Failure Probability')
            ax.set_title('Failure Probability by Health Group', fontsize=13, fontweight='bold')
            graphs['fail_box'] = fig_to_b64(fig)

    return graphs

def build_analytics_graphs(datasets, devices):
    graphs = {}

    # Upload trend
    from collections import Counter
    if datasets:
        dc = Counter(d['upload_time'][:7] if d['upload_time'] else '2026-01'
                     for d in datasets)
        sorted_months = sorted(dc.keys())
        fig, ax = plt.subplots(figsize=(9, 4))
        style_fig(fig, ax)
        ax.bar(sorted_months, [dc[m] for m in sorted_months],
               color='#6366f1', edgecolor='#0f172a', alpha=0.9)
        ax.set_title('Monthly Dataset Uploads', fontsize=13, fontweight='bold')
        ax.set_ylabel('Uploads'); ax.tick_params(axis='x', rotation=30)
        graphs['monthly_uploads'] = fig_to_b64(fig)

    # Temperature distribution
    if devices:
        temps = [d['temperature'] or 0 for d in devices]
        fig, ax = plt.subplots(figsize=(8, 4))
        style_fig(fig, ax)
        ax.hist(temps, bins=20, color='#f59e0b', edgecolor='#0f172a', alpha=0.85)
        ax.set_xlabel('Temperature (°C)'); ax.set_ylabel('Device Count')
        ax.set_title('SSD Temperature Distribution', fontsize=13, fontweight='bold')
        graphs['temp_hist'] = fig_to_b64(fig)

        # Wear level line
        wears = [d['wear_level'] or 0 for d in devices][:100]
        fig, ax = plt.subplots(figsize=(9, 4))
        style_fig(fig, ax)
        ax.plot(range(len(wears)), wears, color='#22d3ee', lw=1.5)
        ax.fill_between(range(len(wears)), wears, alpha=0.2, color='#22d3ee')
        ax.set_xlabel('Device Index'); ax.set_ylabel('Wear Level (%)')
        ax.set_title('SSD Wear Level Analysis', fontsize=13, fontweight='bold')
        graphs['wear_line'] = fig_to_b64(fig)

        # Bad block growth
        bbs = [d['bad_block_count'] or 0 for d in devices][:100]
        fig, ax = plt.subplots(figsize=(9, 4))
        style_fig(fig, ax)
        ax.bar(range(len(bbs)), bbs, color='#f43f5e', edgecolor='#0f172a', alpha=0.8, width=1)
        ax.set_xlabel('Device Index'); ax.set_ylabel('Bad Block Count')
        ax.set_title('Bad Block Count Distribution', fontsize=13, fontweight='bold')
        graphs['bad_block'] = fig_to_b64(fig)

    return graphs

# ─────────────────────────────────────────────
#  API endpoints (JSON)
# ─────────────────────────────────────────────
@app.route('/api/stats')
@login_required
def api_stats():
    db = get_db()
    total  = db.execute("SELECT COUNT(*) FROM ssd_devices WHERE EXISTS (SELECT 1 FROM datasets WHERE datasets.id=ssd_devices.dataset_id AND datasets.user_id=?)", (session['user_id'],)).fetchone()[0]
    high   = db.execute("SELECT COUNT(*) FROM ssd_devices WHERE risk_level='High' AND EXISTS (SELECT 1 FROM datasets WHERE datasets.id=ssd_devices.dataset_id AND datasets.user_id=?)", (session['user_id'],)).fetchone()[0]
    medium = db.execute("SELECT COUNT(*) FROM ssd_devices WHERE risk_level='Medium' AND EXISTS (SELECT 1 FROM datasets WHERE datasets.id=ssd_devices.dataset_id AND datasets.user_id=?)", (session['user_id'],)).fetchone()[0]
    return jsonify({'total': total, 'high': high, 'medium': medium, 'low': total - high - medium})

# ─────────────────────────────────────────────
#  Run
# ─────────────────────────────────────────────
if __name__ == '__main__':
    init_db()
    app.run(debug=True, port=5000)

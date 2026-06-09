"""
modelling.py - MLflow Project Entry Point
==========================================
Script pelatihan model untuk MLflow Project.
"""

import pandas as pd
import numpy as np
import argparse
import mlflow
import mlflow.sklearn
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix
)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ─── Konfigurasi DagsHub ──────────────────────────────────────────────────────
DAGSHUB_OWNER   = 'melindasevira7-bit'
DAGSHUB_REPO    = 'Eksperimen_SML_Melinda-Sevira'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str,
                        default='smsa_preprocessing/smsa_preprocessed.csv')
    parser.add_argument('--max_features', type=int, default=5000)
    parser.add_argument('--ngram_range',  type=str, default='1,2')
    parser.add_argument('--C',            type=float, default=1.0)
    parser.add_argument('--test_size',    type=float, default=0.2)
    parser.add_argument('--random_state', type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()

    # Parse ngram_range
    ngram = tuple(int(x) for x in args.ngram_range.split(','))

    # Init MLflow ke DagsHub
    print("Menginisialisasi DagsHub...")
    token = os.getenv('DAGSHUB_TOKEN', '')
    mlflow.set_tracking_uri(
        f"https://dagshub.com/{DAGSHUB_OWNER}/{DAGSHUB_REPO}.mlflow"
    )
    os.environ['MLFLOW_TRACKING_USERNAME'] = DAGSHUB_OWNER
    os.environ['MLFLOW_TRACKING_PASSWORD'] = token

    # Load data
    print(f"Memuat dataset dari: {args.dataset_path}")
    df = pd.read_csv(args.dataset_path)
    print(f"Shape: {df.shape}")

    # Prepare features
    X = df['text_clean'].fillna('')
    y = df['label_encoded']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size,
        random_state=args.random_state, stratify=y
    )

    # Build pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=args.max_features,
            ngram_range=ngram,
            sublinear_tf=True
        )),
        ('clf', LogisticRegression(
            C=args.C,
            max_iter=1000,
            random_state=args.random_state,
            solver='lbfgs'
        ))
    ])

    import time
with mlflow.start_run(run_name=f"run_{int(time.time())}"):
        # Train
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        # Metrics
        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall    = recall_score(y_test, y_pred, average='weighted')
        f1        = f1_score(y_test, y_pred, average='weighted')

        print(f"Accuracy : {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1-Score : {f1:.4f}")

        # Log params
        mlflow.log_param("max_features", args.max_features)
        mlflow.log_param("ngram_range",  str(ngram))
        mlflow.log_param("C",            args.C)
        mlflow.log_param("test_size",    args.test_size)
        mlflow.log_param("random_state", args.random_state)

        # Log metrics
        mlflow.log_metric("accuracy",           accuracy)
        mlflow.log_metric("precision_weighted", precision)
        mlflow.log_metric("recall_weighted",    recall)
        mlflow.log_metric("f1_weighted",        f1)

        # Log confusion matrix
        cm     = confusion_matrix(y_test, y_pred)
        labels = [str(l) for l in sorted(np.unique(y_test))]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=100)
        plt.close()
        mlflow.log_artifact('confusion_matrix.png')

        # Log classification report
        report = classification_report(y_test, y_pred,
                                       target_names=labels,
                                       output_dict=True)
        with open('classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact('classification_report.json')

        # Log model
        mlflow.sklearn.log_model(pipeline, "model")

        print("Model berhasil disimpan ke DagsHub!")

if __name__ == '__main__':
    main()

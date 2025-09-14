import os
# Riduci i messaggi di log TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import tkinter as tk
from tkinter import messagebox

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

import tensorflow as tf
from tensorflow.keras import Input, Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- File paths ---
BASE = Path(__file__).parent
MODEL_PATH = BASE / "model_best.h5"
SCALER_PATH = BASE / "scaler.joblib"
DATA_PATH = BASE / "dataset.csv"

# --- Se il dataset non esiste, creane uno di esempio ---
if not DATA_PATH.exists():
    df0 = pd.DataFrame({
        'età': [22, 25, 47, 52, 46, 56, 33, 40, 29, 60, 35, 28, 44, 50, 38],
        'stipendio': [1500, 2000, 5000, 6000, 3500, 8000, 4200, 4800, 3000, 9000,
                      4100, 2900, 4600, 5500, 3800],
        'esperienza': [1, 2, 20, 25, 15, 30, 8, 12, 4, 35, 9, 3, 18, 22, 10],
        'compra': [0, 0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0]
    })
    df0.to_csv(DATA_PATH, index=False)

# --- Carica dataset ---
def load_dataset():
    df = pd.read_csv(DATA_PATH)
    X = df[['età', 'stipendio', 'esperienza']].values
    y = df['compra'].values
    return df, X, y

# --- Costruisci modello ---
def build_model(input_dim=3):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(16, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# --- Addestramento ---
def train_model(n_epochs=100, verbose=0, update_status=None):
    df, X, y = load_dataset()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    joblib.dump(scaler, SCALER_PATH)

    model = build_model(input_dim=X.shape[1])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10,
                      restore_best_weights=True, verbose=0),
        ModelCheckpoint(str(MODEL_PATH), monitor='val_loss',
                        save_best_only=True, verbose=0)
    ]

    if update_status:
        update_status("Training in corso...")

    history = model.fit(
        X_train_s, y_train,
        validation_data=(X_val_s, y_val),
        epochs=n_epochs,
        callbacks=callbacks,
        verbose=verbose
    )

    preds = (model.predict(X_val_s, verbose=0) >= 0.5).astype(int).ravel()
    acc = accuracy_score(y_val, preds)

    try:
        auc = roc_auc_score(y_val, model.predict(X_val_s, verbose=0))
    except Exception:
        auc = float('nan')

    if update_status:
        update_status(f"Training completato — acc {acc:.3f}, auc {auc:.3f}")

    return acc, auc

# --- Carica modello e scaler ---
def load_model_and_scaler():
    scaler = joblib.load(SCALER_PATH) if SCALER_PATH.exists() else None
    model = tf.keras.models.load_model(str(MODEL_PATH)) if MODEL_PATH.exists() else None
    return model, scaler

# --- Predici un singolo esempio ---
def predict_single(eta, stipendio, esperienza):
    model, scaler = load_model_and_scaler()
    if model is None or scaler is None:
        raise RuntimeError("Modello o scaler non trovati. Addestra il modello prima (Retrain).")

    X_in = np.array([[eta, stipendio, esperienza]])
    X_s = scaler.transform(X_in)
    prob = float(model.predict(X_s, verbose=0)[0][0])
    return prob

# --- Aggiungi esempio ---
def add_sample_to_dataset(eta, stipendio, esperienza, compra):
    df = pd.read_csv(DATA_PATH)
    new = {'età': eta, 'stipendio': stipendio, 'esperienza': esperienza, 'compra': int(compra)}
    df = pd.concat([df, pd.DataFrame([new])], ignore_index=True)
    df.to_csv(DATA_PATH, index=False)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("AI migliorata - Previsione Acquisto")

tk.Label(root, text="Età:").grid(row=0, column=0)
entry_eta = tk.Entry(root)
entry_eta.grid(row=0, column=1)

tk.Label(root, text="Stipendio:").grid(row=1, column=0)
entry_stipendio = tk.Entry(root)
entry_stipendio.grid(row=1, column=1)

tk.Label(root, text="Esperienza:").grid(row=2, column=0)
entry_esperienza = tk.Entry(root)
entry_esperienza.grid(row=2, column=1)

status_label = tk.Label(root, text="Pronto", fg="blue")
status_label.grid(row=4, column=0, columnspan=2)

def set_status(text, color="blue"):
    status_label.config(text=text, fg=color)
    root.update_idletasks()

def on_predict():
    try:
        eta = float(entry_eta.get())
        stipendio = float(entry_stipendio.get())
        esperienza = float(entry_esperienza.get())
    except ValueError:
        messagebox.showerror("Errore", "Inserisci numeri validi")
        return

    try:
        prob = predict_single(eta, stipendio, esperienza)
    except RuntimeError as e:
        messagebox.showwarning("Modello mancante", str(e))
        return

    risultato = "Sì" if prob >= 0.5 else "No"
    set_status(f"Predizione: {risultato} ({prob*100:.1f}%)", color="green")

def on_retrain():
    set_status("Avviando ri-addestramento...", "orange")
    root.update_idletasks()
    acc, auc = train_model(n_epochs=200, verbose=0, update_status=set_status)
    messagebox.showinfo("Training", f"Completato.\nAccuracy valida: {acc:.3f}\nAUC (approx): {auc:.3f}")

def on_add_sample():
    try:
        eta = float(entry_eta.get())
        stipendio = float(entry_stipendio.get())
        esperienza = float(entry_esperienza.get())
    except ValueError:
        messagebox.showerror("Errore", "Inserisci numeri validi")
        return

    resp = messagebox.askyesno("Aggiungi esempio", "L'utente ha comprato? (Sì = Yes, No = No)")
    add_sample_to_dataset(eta, stipendio, esperienza, compra=resp)
    set_status("Esempio aggiunto al dataset.", "blue")

tk.Button(root, text="Predici", command=on_predict).grid(row=3, column=0)
tk.Button(root, text="Retrain (ri-addestra e salva)", command=on_retrain).grid(row=3, column=1)
tk.Button(root, text="Aggiungi esempio (al dataset)", command=on_add_sample).grid(row=5, column=0, columnspan=2)

root.mainloop()

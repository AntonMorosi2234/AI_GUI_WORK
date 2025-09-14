# AI_GUI_WORK


# 🧠 AI GUI – Previsione Acquisto

Un’applicazione in **Python + Tkinter** che utilizza una **rete neurale Keras** per prevedere se un utente acquisterà un prodotto, sulla base di:

* 👤 **Età**
* 💰 **Stipendio**
* 📈 **Esperienza lavorativa**

Include una GUI (interfaccia grafica) per inserire dati, fare predizioni, aggiungere esempi al dataset e ri-addestrare il modello.

---

## 📂 Struttura del progetto

```
ai_gui_predict/
│
├── dataset.csv        # dataset iniziale o aggiornato
├── model_best.h5      # modello addestrato (generato al primo training)
├── scaler.joblib      # scaler per normalizzare i dati
├── main.py            # script principale con GUI Tkinter
└── README.md          # documentazione
```

---

## ⚙️ Installazione

1. Assicurati di avere **Python 3.9+** installato.
2. Installa le dipendenze necessarie:

```bash
pip install numpy pandas scikit-learn tensorflow joblib
```

Tkinter è già incluso in Python standard.

---

## ▶️ Avvio

Esegui il programma con:

```bash
python main.py
```

Si aprirà una finestra Tkinter.

---

## 🧑‍💻 Utilizzo

* Inserisci i dati nelle caselle:

  * **Età**
  * **Stipendio**
  * **Esperienza (anni)**

* Premi **Predici** → il modello calcolerà la probabilità di acquisto e mostrerà `Sì` o `No`.

* Premi **Retrain (ri-addestra e salva)** → allena la rete neurale sul dataset aggiornato (salva anche `model_best.h5` e `scaler.joblib`).

* Premi **Aggiungi esempio (al dataset)** → aggiungi i dati correnti al dataset, specificando se l’utente ha acquistato o meno.

---

## 📊 Dataset

* Se non esiste `dataset.csv`, al primo avvio ne viene generato uno di esempio con dati fittizi.
* Ogni volta che aggiungi esempi, vengono salvati in `dataset.csv`.
* Il modello utilizza tre feature (`età`, `stipendio`, `esperienza`) per predire la variabile target `compra` (0 = no, 1 = sì).

---

## 🔧 Dettagli tecnici

* **Rete neurale**:

  * 3 input (età, stipendio, esperienza)
  * 2 hidden layer (16 e 8 neuroni, ReLU)
  * 1 output (sigmoide → probabilità)
* **Training**:

  * `adam` optimizer
  * `binary_crossentropy` loss
  * EarlyStopping (pazienza 10)
  * ModelCheckpoint (salva il miglior modello in `model_best.h5`)
* **Metriche**: accuracy + ROC AUC

---

## 📌 Esempio di utilizzo

1. Inserisci:

   ```
   Età: 35
   Stipendio: 4000
   Esperienza: 10
   ```
2. Premi **Predici**
   Output esempio:

   ```
   Predizione: Sì (72.5%)
   ```

---

## 🔮 Idee future

* Aggiungere più feature (es. livello di istruzione, settore lavorativo).
* Salvare e caricare i log del training.
* Visualizzare grafici di training/validazione con matplotlib.
* Supporto a più modelli (es. Random Forest, Logistic Regression).

---


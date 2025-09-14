import pandas as pd
import numpy as np
from pathlib import Path

# Percorso dataset
DATA_PATH = Path(__file__).parent / "dataset.csv"

def generate_synthetic_data(n_samples=200, seed=42):
    np.random.seed(seed)
    
    # Età tra 20 e 65
    eta = np.random.randint(20, 66, size=n_samples)
    # Stipendio tra 1500 e 9000
    stipendio = np.random.randint(1500, 9001, size=n_samples)
    # Esperienza tra 1 e 40
    esperienza = np.random.randint(1, 41, size=n_samples)
    
    # Regola "realistica": più esperienza + stipendio + età → maggiore probabilità di comprare
    score = (
        0.02 * (eta - 20) +
        0.0004 * stipendio +
        0.05 * esperienza +
        np.random.normal(0, 1, n_samples)  # un po' di rumore
    )
    
    # Trasformo in probabilità (sigmoid)
    prob = 1 / (1 + np.exp(-score/10))
    
    # Genero target binario
    compra = (prob > 0.5).astype(int)
    
    # Creo DataFrame
    df = pd.DataFrame({
        "età": eta,
        "stipendio": stipendio,
        "esperienza": esperienza,
        "compra": compra
    })
    return df

def main():
    print("🔄 Generazione dataset sintetico...")
    df = generate_synthetic_data(n_samples=200)
    
    if DATA_PATH.exists():
        print("📂 Dataset trovato, concateno con quello esistente.")
        df0 = pd.read_csv(DATA_PATH)
        df = pd.concat([df0, df], ignore_index=True)
    
    # Salva dataset aggiornato
    df.to_csv(DATA_PATH, index=False)
    print(f"✅ Dataset salvato in {DATA_PATH} ({len(df)} righe totali)")

if __name__ == "__main__":
    main()

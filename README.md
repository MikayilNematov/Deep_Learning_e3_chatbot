# RAG-baserad chattbot för E3 Control

Detta projekt är en del av inlämningsuppgiften: Implementera en chattbot med RAG. Målet är att skapa en kontextmedveten chattbot som använder RAG-teknik för att besvara frågor om E3 Control – Ett automationsföretag som producerar elcentraler, effektbrytare m.m.

--------------------------------------------------------------------------------------------

# Teknikstack

- **Språkmodell:** `all-MiniLM-L6-v2` via `sentence-transformers`
- **Vektorindex:** FAISS
- **Gränssnitt:** Streamlit
- **RAG:** Retrieval-Augmented Generation via FAQ-matchning + indexsökning

--------------------------------------------------------------------------------------------

# Projektstruktur

```
├── app.py                  # Streamlit chattbot
├── index.ipynb             # Skapar FAISS-index
├── Data/
│   ├── text_entries.txt    # Innehåll för kontextbaserad sökning
│   └── faiss_index.index   # FAISS-index
├── images.jpg              # Bakgrundsbild
```

--------------------------------------------------------------------------------------------

# Så här kör du projektet

# 1. Installera beroenden

```bash
pip install -r requirements.txt
```

Exempel på nödvändiga paket:

```text
streamlit
sentence-transformers
faiss-cpu
scikit-learn
geopy
openrouteservice
```

# 2. Skapa index (endast första gången)

Kör `index.ipynb` för att:
- Läsa in `text_entries.txt`
- Generera embeddings
- Spara ner FAISS-index

# 3. Starta chattboten

```bash
streamlit run app.py
```

--------------------------------------------------------------------------------------------

# Användningsområden

Den här chattboten kan användas av:

- **Supportteamet** på E3 Control för att effektivt besvara vanliga frågor
- **Kunder** som vill hitta rätt manual eller ställa tekniska frågor om effektbrytare
- **Säljare** som snabbt vill ge prisuppgifter eller fraktinformation

--------------------------------------------------------------------------------------------

# Exempel på funktioner

- Semantisk matchning mot 80+ FAQ-frågor
- Dynamiska länkar till tekniska dokument
- Fraktkostnadsberäkning baserat på användarens adress
- Anpassade svar beroende på frågetyp (Micrologic, beställning, leverans m.m.)

--------------------------------------------------------------------------------------------
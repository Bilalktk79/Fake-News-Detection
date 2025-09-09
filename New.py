import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import re
import string
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
import sqlite3
import warnings
warnings.filterwarnings('ignore')

# 1. Load and Merge Dataset
def load_data():
    fake = pd.read_csv("Fake.csv")
    true = pd.read_csv("True.csv")
    fake["label"] = 1
    true["label"] = 0
    df = pd.concat([fake, true], axis=0)
    df = df[["text", "label"]]
    return df

# 2. Clean Text
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\@w+|\#','', text)
    text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# 3. Train Models and Save Best One
def train_model():
    df = load_data()
    df['text'] = df['text'].apply(clean_text)
    X = df['text']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Naive Bayes": MultinomialNB()
    }

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc = accuracy_score(y_test, preds)
        trained_models[name] = (model, acc)

    best_model = max(trained_models.items(), key=lambda x: x[1][1])[0]
    with open("fake_news_model.pkl", "wb") as f:
        pickle.dump(trained_models[best_model][0], f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    return trained_models, best_model

# Create SQLite database to log inputs
def setup_db():
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS results (news TEXT, prediction TEXT, is_correct TEXT DEFAULT NULL)''')
    c.execute('''CREATE TABLE IF NOT EXISTS suggestions (id INTEGER PRIMARY KEY AUTOINCREMENT, suggestion TEXT)''')
    conn.commit()
    conn.close()

def log_prediction(news, result):
    conn = sqlite3.connect("predictions.db")
    c = conn.cursor()
    c.execute("INSERT INTO results (news, prediction) VALUES (?, ?)", (news, result))
    conn.commit()
    conn.close()

# Train Model
train_model()
setup_db()

# Load model and vectorizer
with open("fake_news_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# GUI using Tkinter
def predict():
    news_input = entry.get("1.0", tk.END).strip()
    if news_input == "":
        messagebox.showwarning("Input Error", "Please enter some news text.")
        return
    cleaned = clean_text(news_input)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    result = "❗ FAKE NEWS" if prediction == 1 else "✅ REAL NEWS"
    result_label.config(text=f"Prediction: {result}")
    log_prediction(news_input, result)

def mark_correctness():
    last_news = entry.get("1.0", tk.END).strip()
    if last_news:
        correct = messagebox.askyesno("Confirmation", "Was the prediction correct?")
        conn = sqlite3.connect("predictions.db")
        c = conn.cursor()
        c.execute("UPDATE results SET is_correct = ? WHERE news = ?", ("Yes" if correct else "No", last_news))
        conn.commit()
        conn.close()
        messagebox.showinfo("Saved", "Thanks! Your feedback is recorded.")

def submit_suggestion():
    suggestion = simpledialog.askstring("Suggestion", "Enter your suggestion:")
    if suggestion:
        conn = sqlite3.connect("predictions.db")
        c = conn.cursor()
        c.execute("INSERT INTO suggestions (suggestion) VALUES (?)", (suggestion,))
        conn.commit()
        conn.close()
        messagebox.showinfo("Saved", "Your suggestion has been saved.")

def upload_text():
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            entry.delete("1.0", tk.END)
            entry.insert(tk.END, content)

def clear_text():
    entry.delete("1.0", tk.END)
    result_label.config(text="Prediction:")

app = tk.Tk()
app.title("Fake News Detection")
app.geometry("700x600")

label = tk.Label(app, text="Enter News Text:", font=("Arial", 14))
label.pack(pady=10)

entry = tk.Text(app, height=10, width=80)
entry.pack(pady=5)

button_frame = tk.Frame(app)
button_frame.pack(pady=10)

tk.Button(button_frame, text="Predict", command=predict, font=("Arial", 12), bg="blue", fg="white").grid(row=0, column=0, padx=5)
tk.Button(button_frame, text="Clear", command=clear_text, font=("Arial", 12)).grid(row=0, column=1, padx=5)
tk.Button(button_frame, text="Upload File", command=upload_text, font=("Arial", 12)).grid(row=0, column=2, padx=5)
tk.Button(button_frame, text="Mark Prediction", command=mark_correctness, font=("Arial", 12)).grid(row=0, column=3, padx=5)
tk.Button(button_frame, text="Add Suggestion", command=submit_suggestion, font=("Arial", 12)).grid(row=0, column=4, padx=5)

result_label = tk.Label(app, text="Prediction:", font=("Arial", 14))
result_label.pack(pady=10)

app.mainloop()

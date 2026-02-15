import numpy as np

import re
import time

from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *

class SportsPoliticsClassifier:
   

    def __init__(self, random_seed=42):
        self.seed = random_seed
        self.vectorizer = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        
        # Defining the newsgroup categories to fetch
        self.target_categories = [
            'rec.sport.baseball', 'rec.sport.hockey',   # for sports
            'talk.politics.mideast', 'talk.politics.misc', 'talk.politics.guns' # for politics
        ]
        
        # Dictionary to store performance metrics for the 5-page report
        self.results_summary = {}

    def text_parser(self, text):
        """
        Custom parser to clean raw text. 
        Mirrors the logic of the SMS parser in the reference code.
        """
        # Remove non-letters and convert to lowercase
        clean_pattern = re.compile(r'[^a-zA-Z]')
        words = clean_pattern.split(text)
        
        # Filter out empty strings and single characters
        cleaned_tokens = [w.lower() for w in words if len(w) > 1]
        return " ".join(cleaned_tokens)

    def load_and_preprocess(self):
        """
        Loads the dataset and converts sub-categories into binary labels.
        """
        print("[INFO] Initializing data collection from 20 Newsgroups...")
        
        # Fetching data with metadata removal to ensure valid learning
        raw_data = fetch_20newsgroups(
            subset='all', 
            categories=self.target_categories,
            remove=('headers', 'footers', 'quotes')
        )

        processed_texts = []
        binary_labels = []

        print(f"[INFO] Processing {len(raw_data.data)} documents...")
        
        for i in range(len(raw_data.data)):
            # Manual preprocessing step
            clean_text = self.text_parser(raw_data.data[i])
            processed_texts.append(clean_text)
            
            # Label mapping: indices 0,1 are Sports; 2,3,4 are Politics
            if raw_data.target[i] < 2:
                binary_labels.append(0) # Sports
            else:
                binary_labels.append(1) # Politics
                
        return processed_texts, np.array(binary_labels)

    def feature_engineering(self, texts, labels):
        """
        Converts text to TF-IDF representation as required by the problem.
        Uses bigrams to capture more context for the classifier.
        """
        print("[INFO] Vectorizing features using TF-IDF (N-grams: 1 to 2)...")
        
        # Initializing the TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=5000,
            ngram_range=(1, 2)
        )
        
        X_tfidf = self.vectorizer.fit_transform(texts)
        
        # Performing the 80/20 train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_tfidf, labels, test_size=0.2, random_state=self.seed
        )
        
        print(f"[SUCCESS] Feature matrix shape: {X_tfidf.shape}")

    def evaluate_model(self, name, model):
        """
        Standardized function to train and evaluate each ML technique.
        """
        start_time = time.time()
        print(f"\n[RUNNING] Starting evaluation for: {name}...")
        
        # Training the classifier
        model.fit(self.X_train, self.y_train)
        
        # Generating predictions for quantitative comparison
        preds = model.predict(self.X_test)
        training_time = time.time() - start_time
        
        # Calculating scores for the final report tables
        acc = accuracy_score(self.y_test, preds)
        self.results_summary[name] = {
            'accuracy': acc,
            'report': classification_report(self.y_test, preds, target_names=['Sports', 'Politics']),
            'time': training_time
        }
        
        print(f"[DONE] {name} Accuracy: {acc:.4f} (Time: {training_time:.2f}s)")

    def display_comparative_analysis(self):
        """
        Outputs the final quantitative data for the report.
        """
        print("\n" + "="*60)
        print("          FINAL QUANTITATIVE COMPARISON TABLE")
        print("="*60)
        print(f"{'ML Technique':<35} | {'Accuracy':<10} | {'Train Time':<10}")
        print("-" * 60)
        
        for name, stats in self.results_summary.items():
            print(f"{name:<35} | {stats['accuracy']:<10.2%} | {stats['time']:<10.3f}s")
            
        print("\n[INFO] Detailed Classification Reports:")
        for name, stats in self.results_summary.items():
            print(f"\n--- {name} ---")
            print(stats['report'])

    def run(self):
        """
        Main execution flow.
        """
        # Step 1: Data Work
        texts, labels = self.load_and_preprocess()
        
        # Step 2: Feature Work
        self.feature_engineering(texts, labels)
        
        # Step 3: Technique Comparison
        # technique 1: Naive Bayes (Probabilistic)
        self.evaluate_model("Multinomial Naive Bayes", MultinomialNB())
        
        # technique 2: SVM (Linear Kernel)
        self.evaluate_model("Support Vector Machine (Linear)", SVC(kernel='linear'))
        
        # technique 3: Logistic Regression (Discriminative)
        self.evaluate_model("Logistic Regression", LogisticRegression(max_iter=1000))
        
        # Step 4: Final Output
        self.display_comparative_analysis()

if __name__ == "__main__":
    # to run the pipeline
    project = SportsPoliticsClassifier(random_seed=42)
    project.run()
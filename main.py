import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, chi2
import re
import string
from scipy.sparse import hstack, csr_matrix
import warnings
warnings.filterwarnings('ignore')

# Due to nltk import issues
ENGLISH_STOP_WORDS = {
    'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", 
    "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 
    'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 
    'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 
    'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 
    'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 
    'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 
    'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'through', 'during', 
    'before', 'after', 'above', 'below', 'up', 'down', 'in', 'out', 'on', 'off', 
    'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 
    'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 
    'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
    'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 
    'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 
    'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 
    'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 
    'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 
    'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 
    'wouldn', "wouldn't",
    
    # Political-specific stop words
    'said', 'says', 'according', 'statement', 'claim', 'claims', 'politician', 
    'government', 'administration', 'party', 'official', 'spokesperson'
}
def load_dataset():
        columns = [
            'id', 'label', 'statement', 'subject', 'speaker',
            'job_title', 'state_info', 'party_affiliation',
            'barely_true_counts', 'false_counts', 'half_true_counts',
            'mostly_true_counts', 'pants_on_fire_counts', 'context'
        ]
        
        datasets = {}
        fileNames = ['train.tsv', 'test.tsv', 'valid.tsv']
        
        for fileName in fileNames:
            df = pd.read_csv(fileName, sep='\t', header=None, names=columns)
            datasets[fileName] = df
        
        if not datasets:
            raise Exception("No TSV files found")

        df = pd.concat(list(datasets.values()), ignore_index=True)

        df = df.dropna(subset=['statement', 'label'])
        df['statement'] = df['statement'].astype(str)
        df['party'] = df['party_affiliation'].fillna('unknown')
        
        return df
    
# Due to nltk import issues
class AdvancedTextPreprocessor:
    def __init__(self):
        self.stop_words = ENGLISH_STOP_WORDS
    
    def stem(self, word):
        if len(word) <= 3:
            return word
        if word.endswith('ies'):
            return word[:-3] + 'y'
        elif word.endswith('ied'):
            return word[:-3] + 'y'
        elif word.endswith('ing'):
            return word[:-3]
        elif word.endswith('ly'):
            return word[:-2]
        elif word.endswith('ed'):
            return word[:-2]
        elif word.endswith('s') and not word.endswith('ss') and not word.endswith('us'):
            return word[:-1]
        elif word.endswith('er'):
            return word[:-2]
        elif word.endswith('est'):
            return word[:-3]
        
        return word

class model:
    
    def __init__(self):
        self.preprocessor = AdvancedTextPreprocessor()
        self.models = {}
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.stop_words = ENGLISH_STOP_WORDS

    def advancedClean(self, text):
        text = text.lower()
        text = re.sub(r'https?://\S+', '', text)
        text = re.sub(r'\d+', '', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def createAdvancedFeatures(self, texts):
        tfidf = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            min_df=3,
            max_df=0.95,
            stop_words=list(self.stop_words),
            sublinear_tf=True,
            strip_accents='unicode'
        )
        
        cleanedTexts = [self.advancedClean(text) for text in texts]

        tfidfFeatures = tfidf.fit_transform(cleanedTexts)

        linguisticFeatures = []
        
        for text in texts:
            features = self.extractLinguisticFeatures(text)
            if features:
                linguisticFeatures.append(list(features.values()))
            else:
                linguisticFeatures.append([0] * 10)
        
        linguisticFeatures = np.array(linguisticFeatures)
        
        if linguisticFeatures.size > 0 and linguisticFeatures.shape[0] > 0:
            linguisticSparse = csr_matrix(linguisticFeatures)
            combinedFeatures = hstack([tfidfFeatures, linguisticSparse])
        else:
            combinedFeatures = tfidfFeatures

        self.tfidf_vectorizer = tfidf

        tfidfNames = list(tfidf.get_feature_names_out())
        linguisticNames = [
            'uncertainty_ratio', 'strong_assertion_ratio', 'emotional_ratio', 
            'superlative_ratio', 'avg_word_length', 'sentence_count',
            'question_marks', 'exclamation_marks', 'capitalized_words', 'number_mentions'
        ]
        self.feature_names = tfidfNames + linguisticNames
        
        return combinedFeatures

    def tokenize(self, text):
        text = re.sub(r'[^\w\s]', ' ', text)
        return [word.strip() for word in text.split() if word.strip()]
    
    def extractLinguisticFeatures(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return {}
        
        textLowered = text.lower()
        words = self.tokenize(text)
        
        uncertainty_words = [
            'maybe', 'probably', 'might', 'could', 'possibly', 'perhaps',
            'allegedly', 'reportedly', 'supposedly', 'seems', 'appears',
            'suggests', 'indicates', 'may', 'would', 'should'
        ]

        strong_words = [
            'definitely', 'certainly', 'absolutely', 'completely', 'totally',
            'never', 'always', 'all', 'every', 'none', 'must', 'will'
        ]

        emotional_words = [
            'terrible', 'awful', 'amazing', 'incredible', 'fantastic',
            'horrible', 'wonderful', 'devastating', 'shocking', 'outrageous'
        ]

        superlatives = [
            'best', 'worst', 'greatest', 'largest', 'smallest', 'highest',
            'lowest', 'most', 'least', 'first', 'last', 'biggest'
        ]

        word_count = len(words)
        if word_count == 0:
            return {}
        
        features = {
            'uncertainty_ratio': sum(1 for word in uncertainty_words if word in textLowered) / word_count,
            'strong_assertion_ratio': sum(1 for word in strong_words if word in textLowered) / word_count,
            'emotional_ratio': sum(1 for word in emotional_words if word in textLowered) / word_count,
            'superlative_ratio': sum(1 for word in superlatives if word in textLowered) / word_count,
            'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'question_marks': text.count('?'),
            'exclamation_marks': text.count('!'),
            'capitalized_words': sum(1 for word in words if word.isupper() and len(word) > 1),
            'number_mentions': len(re.findall(r'\d+', text))
        }
        
        return features
        
    def train_high_performance_models(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        results = {}
        
        # Model 1: Logistic Regression
        lr_pipeline = Pipeline([
            ('feature_selection', SelectKBest(chi2, k=5000)),
            ('scaler', StandardScaler(with_mean=False)),  # For sparse matrices
            ('classifier', LogisticRegression(
                class_weight='balanced',
                max_iter=2000,
                C=1.0,
                solver='liblinear'
            ))
        ])
        
        lr_pipeline.fit(X_train, y_train)
        lr_pred = lr_pipeline.predict(X_test)
        
        results['Optimized Logistic Regression'] = {
            'model': lr_pipeline,
            'predictions': lr_pred,
            'f1_macro': f1_score(y_test, lr_pred, average='macro'),
            'accuracy': accuracy_score(y_test, lr_pred)
        }
        
        # Model 2: Support Vector Machine
        svm_pipeline = Pipeline([
            ('feature_selection', SelectKBest(chi2, k=3000)),
            ('scaler', StandardScaler(with_mean=False)),
            ('classifier', SVC(
                class_weight='balanced',
                kernel='linear',
                C=0.1,
                random_state=42
            ))
        ])
        
        svm_pipeline.fit(X_train, y_train)
        svm_pred = svm_pipeline.predict(X_test)
        
        results['Support Vector Machine'] = {
            'model': svm_pipeline,
            'predictions': svm_pred,
            'f1_macro': f1_score(y_test, svm_pred, average='macro'),
            'accuracy': accuracy_score(y_test, svm_pred)
        }
        
        # Model 3: Gradient Boosting
        gb_pipeline = Pipeline([
            ('feature_selection', SelectKBest(chi2, k=2000)),
            ('classifier', GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            ))
        ])
        
        gb_pipeline.fit(X_train, y_train)
        gb_pred = gb_pipeline.predict(X_test)
        
        results['Gradient Boosting'] = {
            'model': gb_pipeline,
            'predictions': gb_pred,
            'f1_macro': f1_score(y_test, gb_pred, average='macro'),
            'accuracy': accuracy_score(y_test, gb_pred)
        }
        
        # Model 4: Random Forest
        rf_pipeline = Pipeline([
            ('feature_selection', SelectKBest(chi2, k=4000)),
            ('classifier', RandomForestClassifier(
                n_estimators=200,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                class_weight='balanced',
                random_state=42
            ))
        ])
        
        rf_pipeline.fit(X_train, y_train)
        rf_pred = rf_pipeline.predict(X_test)
        
        results['Advanced Random Forest'] = {
            'model': rf_pipeline,
            'predictions': rf_pred,
            'f1_macro': f1_score(y_test, rf_pred, average='macro'),
            'accuracy': accuracy_score(y_test, rf_pred)
        }
        
        # Model 5: Smart Ensemble
        best_models = sorted(results.items(), key=lambda x: x[1]['f1_macro'], reverse=True)[:3]
        
        ensemble_predictions = []
        for i in range(len(y_test)):
            votes = {}
            for model_name, data in best_models:
                pred = data['predictions'][i]
                weight = data['f1_macro']
                if pred not in votes:
                    votes[pred] = 0
                votes[pred] += weight
            best_pred = max(votes.keys(), key=lambda k: votes[k])
            ensemble_predictions.append(best_pred)
        
        ensemble_predictions = np.array(ensemble_predictions)
        
        results['Smart Ensemble'] = {
            'predictions': ensemble_predictions,
            'f1_macro': f1_score(y_test, ensemble_predictions, average='macro'),
            'accuracy': accuracy_score(y_test, ensemble_predictions)
        }
        
        self.models = results
        return results, y_test

def visualizations(results, y_test, le):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Model Performance Comparison
    models = list(results.keys())
    f1_scores = [results[model]['f1_macro'] for model in models]
    accuracies = [results[model]['accuracy'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, f1_scores, width, label='F1-Score (Macro)', alpha=0.8, color='skyblue')
    bars2 = ax1.bar(x + width/2, accuracies, width, label='Accuracy', alpha=0.8, color='lightcoral')
    
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Performance')
    ax1.set_title('Model Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, max(max(f1_scores), max(accuracies)) + 0.1)
    
    # Best Model's Confusion Matrix
    best_model = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    best_predictions = results[best_model]['predictions']
    
    cm = confusion_matrix(y_test, best_predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax2,
               xticklabels=le.classes_, yticklabels=le.classes_)
    ax2.set_title(f'Confusion Matrix - {best_model}\nF1: {results[best_model]["f1_macro"]:.3f}', 
                 fontsize=14, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    # Performance Improvement Chart
    baseline_f1 = 0.167
    improvements = [(results[model]['f1_macro'] - baseline_f1) / baseline_f1 * 100 for model in models]
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
    bars = ax3.bar(range(len(models)), improvements, color=colors, alpha=0.8)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    ax3.set_xlabel('Models')
    ax3.set_ylabel('Improvement over Random (%)')
    ax3.set_title('Performance Improvement Analysis', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Class-wise Performance (via F1-scores)
    best_pred = results[best_model]['predictions']
    class_f1s = f1_score(y_test, best_pred, average=None)
    
    ax4.bar(range(len(le.classes_)), class_f1s, color='lightgreen', alpha=0.8)
    ax4.set_xlabel('Truthfulness Classes')
    ax4.set_ylabel('F1-Score')
    ax4.set_title(f'Class-wise Performance - {best_model}', fontsize=14, fontweight='bold')
    ax4.set_xticks(range(len(le.classes_)))
    ax4.set_xticklabels(le.classes_, rotation=45)
    ax4.grid(True, alpha=0.3)

    for i, v in enumerate(class_f1s):
        ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Summary
    best_f1 = results[best_model]['f1_macro']
    improvement = (best_f1 - baseline_f1) / baseline_f1 * 100
    
    print(f"\n BEST MODEL: {best_model}")
    print(f"   F1-Score: {best_f1:.4f}")
    print(f"   Accuracy: {results[best_model]['accuracy']:.4f}")
    print(f"   Improvement over random: {improvement:.1f}%")
    
    print(f"\n RESULTS:")
    for model, metrics in results.items():
        improvement = (metrics['f1_macro'] - baseline_f1) / baseline_f1 * 100
        print(f"   {model:25s} | F1: {metrics['f1_macro']:.3f} | Acc: {metrics['accuracy']:.3f} | +{improvement:.1f}%")

def runPerformanceAnalysis():
    df = load_dataset()

    le = LabelEncoder()
    y = le.fit_transform(df['label'])
    
    print(f"\n Dataset Summary:")
    print(f" Total statements: {len(df)}")
    print(f" Classes: {len(le.classes_)}")
    print(f"Class Distribution:")
    classDistribution = dict(zip(le.classes_, np.bincount(y)))
    for className, count in classDistribution.items():
        percentage = (count / len(y)) * 100
        print(f"   {className:15s} | {count:4d} samples ({percentage:5.1f}%)")
    
    models = model()
    
    X = models.createAdvancedFeatures(df['statement'])

    results, y_test = models.train_high_performance_models(X, y)
    
    visualizations(results, y_test, le)

runPerformanceAnalysis()
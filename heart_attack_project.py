"""
Heart Attack Prediction using Machine Learning
Complete implementation with data preprocessing, EDA, and model training
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report,
                            roc_auc_score, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

class HeartAttackPredictor:
    def __init__(self, data_path):
        """Initialize the predictor with data path"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def load_data(self):
        """Load the heart disease dataset"""
        # For demonstration, creating sample data structure
        # Replace this with: self.df = pd.read_csv(self.data_path)
        
        print("Loading heart disease dataset...")
        # Sample dataset structure - replace with actual CSV loading
        self.df = pd.DataFrame({
            'age': np.random.randint(30, 80, 303),
            'sex': np.random.randint(0, 2, 303),
            'cp': np.random.randint(0, 4, 303),
            'trestbps': np.random.randint(90, 200, 303),
            'chol': np.random.randint(120, 400, 303),
            'fbs': np.random.randint(0, 2, 303),
            'restecg': np.random.randint(0, 3, 303),
            'thalach': np.random.randint(70, 200, 303),
            'exang': np.random.randint(0, 2, 303),
            'oldpeak': np.random.uniform(0, 6, 303),
            'slope': np.random.randint(0, 3, 303),
            'ca': np.random.randint(0, 4, 303),
            'thal': np.random.randint(0, 4, 303),
            'target': np.random.randint(0, 2, 303)
        })
        
        print(f"Dataset loaded successfully! Shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Perform exploratory data analysis"""
        print("\n" + "="*50)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*50)
        
        # Basic information
        print("\nDataset Info:")
        print(self.df.info())
        
        print("\nDataset Description:")
        print(self.df.describe())
        
        print("\nMissing Values:")
        print(self.df.isnull().sum())
        
        print("\nTarget Distribution:")
        print(self.df['target'].value_counts())
        
        # Visualizations
        self._plot_distributions()
        self._plot_correlations()
        
    def _plot_distributions(self):
        """Plot feature distributions"""
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        axes = axes.ravel()
        
        for i, col in enumerate(self.df.columns):
            if i < 15:
                axes[i].hist(self.df[col], bins=30, edgecolor='black', alpha=0.7)
                axes[i].set_title(f'Distribution of {col}')
                axes[i].set_xlabel(col)
                axes[i].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig('feature_distributions.png', dpi=300, bbox_inches='tight')
        print("\nFeature distributions saved as 'feature_distributions.png'")
        plt.close()
        
    def _plot_correlations(self):
        """Plot correlation matrix"""
        plt.figure(figsize=(14, 10))
        correlation_matrix = self.df.corr()
        sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1)
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight')
        print("Correlation matrix saved as 'correlation_matrix.png'")
        plt.close()
        
    def preprocess_data(self):
        """Preprocess the data for modeling"""
        print("\n" + "="*50)
        print("DATA PREPROCESSING")
        print("="*50)
        
        # Handle missing values
        self.df = self.df.dropna()
        print(f"\nData shape after removing missing values: {self.df.shape}")
        
        # Remove outliers using IQR method
        Q1 = self.df.quantile(0.25)
        Q3 = self.df.quantile(0.75)
        IQR = Q3 - Q1
        self.df = self.df[~((self.df < (Q1 - 1.5 * IQR)) | 
                           (self.df > (Q3 + 1.5 * IQR))).any(axis=1)]
        print(f"Data shape after removing outliers: {self.df.shape}")
        
        # Feature engineering
        self.df['age_group'] = pd.cut(self.df['age'], bins=[0, 40, 55, 70, 100],
                                      labels=[0, 1, 2, 3])
        self.df['age_group'] = self.df['age_group'].astype(int)
        
        # Separate features and target
        X = self.df.drop('target', axis=1)
        y = self.df['target']
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTraining set size: {self.X_train.shape[0]}")
        print(f"Testing set size: {self.X_test.shape[0]}")
        
    def train_models(self):
        """Train multiple machine learning models"""
        print("\n" + "="*50)
        print("MODEL TRAINING")
        print("="*50)
        
        # Define models
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, 
                                    eval_metric='logloss')
        }
        
        # Train and evaluate each model
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred = model.predict(self.X_test)
            y_pred_proba = model.predict_proba(self.X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred)
            recall = recall_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred)
            auc_roc = roc_auc_score(self.y_test, y_pred_proba)
            
            # Cross-validation score
            cv_scores = cross_val_score(model, self.X_train, self.y_train, 
                                       cv=5, scoring='accuracy')
            
            self.results[name] = {
                'model': model,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc_roc': auc_roc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba
            }
            
            print(f"{name} - Accuracy: {accuracy:.4f}")
            print(f"{name} - AUC-ROC: {auc_roc:.4f}")
            print(f"{name} - CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    def evaluate_models(self):
        """Evaluate and compare all models"""
        print("\n" + "="*50)
        print("MODEL EVALUATION")
        print("="*50)
        
        # Create comparison dataframe
        comparison_data = []
        for name, results in self.results.items():
            comparison_data.append({
                'Model': name,
                'Accuracy': f"{results['accuracy']:.4f}",
                'Precision': f"{results['precision']:.4f}",
                'Recall': f"{results['recall']:.4f}",
                'F1-Score': f"{results['f1_score']:.4f}",
                'AUC-ROC': f"{results['auc_roc']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n" + comparison_df.to_string(index=False))
        
        # Plot comparison
        self._plot_model_comparison()
        self._plot_confusion_matrices()
        self._plot_roc_curves()
        
        # Find best model
        best_model_name = max(self.results.keys(), 
                            key=lambda x: self.results[x]['accuracy'])
        print(f"\n🏆 Best Model: {best_model_name}")
        print(f"Accuracy: {self.results[best_model_name]['accuracy']:.4f}")
        
    def _plot_model_comparison(self):
        """Plot model comparison chart"""
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_roc']
        models = list(self.results.keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model in enumerate(models):
            values = [self.results[model][metric] for metric in metrics]
            ax.bar(x + i*width, values, width, label=model)
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x + width)
        ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'])
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
        print("\nModel comparison saved as 'model_comparison.png'")
        plt.close()
        
    def _plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (name, results) in enumerate(self.results.items()):
            cm = confusion_matrix(self.y_test, results['y_pred'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx])
            axes[idx].set_title(f'{name}\nConfusion Matrix', fontweight='bold')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('confusion_matrices.png', dpi=300, bbox_inches='tight')
        print("Confusion matrices saved as 'confusion_matrices.png'")
        plt.close()
        
    def _plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        for name, results in self.results.items():
            fpr, tpr, _ = roc_curve(self.y_test, results['y_pred_proba'])
            auc = results['auc_roc']
            plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.4f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curves Comparison', fontsize=14, fontweight='bold')
        plt.legend(loc='lower right')
        plt.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300, bbox_inches='tight')
        print("ROC curves saved as 'roc_curves.png'")
        plt.close()
    
    def feature_importance(self):
        """Display feature importance for tree-based models"""
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        
        # Get feature names
        feature_names = self.df.drop('target', axis=1).columns
        
        # Plot for Random Forest and XGBoost
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        for idx, name in enumerate(['Random Forest', 'XGBoost']):
            if name in self.results:
                model = self.results[name]['model']
                importances = model.feature_importances_
                indices = np.argsort(importances)[::-1][:10]
                
                axes[idx].barh(range(10), importances[indices], color='steelblue')
                axes[idx].set_yticks(range(10))
                axes[idx].set_yticklabels([feature_names[i] for i in indices])
                axes[idx].set_xlabel('Importance Score', fontweight='bold')
                axes[idx].set_title(f'{name}\nTop 10 Important Features', 
                                   fontweight='bold')
                axes[idx].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
        print("\nFeature importance saved as 'feature_importance.png'")
        plt.close()


def main():
    """Main execution function"""
    print("="*60)
    print("HEART ATTACK PREDICTION USING MACHINE LEARNING")
    print("="*60)
    
    # Initialize predictor
    # Replace 'heart_disease.csv' with your actual dataset path
    predictor = HeartAttackPredictor('heart_disease.csv')
    
    # Load and explore data
    predictor.load_data()
    predictor.explore_data()
    
    # Preprocess data
    predictor.preprocess_data()
    
    # Train models
    predictor.train_models()
    
    # Evaluate models
    predictor.evaluate_models()
    
    # Feature importance
    predictor.feature_importance()
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE! Check the generated visualizations.")
    print("="*60)


if __name__ == "__main__":
    main()

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectFromModel
from sklearn.externals import joblib

# Sample or reduce data size
# Assuming 'data' is your dataset and 'label' is your target variable
# You can modify this section based on how you're loading your data.
sampled_data = data.sample(frac=0.1, random_state=42)  # Use 10% of the data for faster runs
X = sampled_data['text']
y = sampled_data['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorization using TF-IDF (cache this to avoid re-computing in future)
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features for efficiency
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Cache vectorizer for future use
joblib.dump(vectorizer, 'vectorizer.pkl')

# Logistic Regression Model (as a simpler baseline)
lr_model = LogisticRegression(max_iter=100, n_jobs=-1)  # Using parallel processing with n_jobs=-1
lr_model.fit(X_train_vec, y_train)
pred_lr = lr_model.predict(X_test_vec)
print("Logistic Regression Results:\n", classification_report(y_test, pred_lr))

# Feature Selection using L1 regularization to reduce complexity
selector = SelectFromModel(estimator=LogisticRegression(penalty='l1', solver='liblinear')).fit(X_train_vec, y_train)
X_train_selected = selector.transform(X_train_vec)
X_test_selected = selector.transform(X_test_vec)

# Random Forest Classifier with reduced estimators
rf_model = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)  # Parallel processing
rf_model.fit(X_train_selected, y_train)
pred_rf = rf_model.predict(X_test_selected)
print("Random Forest Results:\n", classification_report(y_test, pred_rf))

# Gradient Boosting Classifier with reduced estimators
gb_model = GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, random_state=42)
gb_model.fit(X_train_selected, y_train)
pred_gb = gb_model.predict(X_test_selected)
print("Gradient Boosting Results:\n", classification_report(y_test, pred_gb))

# Manual Testing function (unchanged from your original code)
def output_label(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def manual_testing(news):
    testing_news = {"text": [news]}
    new_def_test = pd.DataFrame(testing_news)
    new_def_test["text"] = new_def_test["text"].apply(wordopt)
    new_x_test = new_def_test["text"]
    new_xv_test = vectorizer.transform(new_x_test)
    new_xv_test_selected = selector.transform(new_xv_test)
    
    pred_LR = lr_model.predict(new_xv_test_selected)
    pred_RF = rf_model.predict(new_xv_test_selected)
    pred_GB = gb_model.predict(new_xv_test_selected)
    
    return print(f"\nLR Prediction: {output_label(pred_LR[0])} "
                 f"\nRFC Prediction: {output_label(pred_RF[0])} "
                 f"\nGBC Prediction: {output_label(pred_GB[0])}")

# Example usage of manual testing
news = str(input("Enter news text: "))
manual_testing(news)

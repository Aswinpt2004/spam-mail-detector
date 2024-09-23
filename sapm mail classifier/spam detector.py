import pandas as pd 
data = pd.read_excel(r"D:\naive bayes spam mail classifier\sapm mail classifier\spam_emails.xlsx")
# handle any potential encoding issues.
data
print(data.head)
import re

# Ensure all text values are strings
data['text'] = data['text'].fillna('').astype(str)

# Preprocess function: lowercase and remove non-alphabetic characters
def preprocess(text):
    return re.sub(r'\W+', ' ', text).lower().split()

# Apply preprocessing
data['text'] = data['text'].apply(preprocess)
data['Spam']=data['label'].apply(lambda x:1 if x=='spam' else 0)
data.head(5)
from sklearn.model_selection import train_test_split

X = data['text']
y = data['Spam']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
def calculate_prior(y_train):
    total = len(y_train)
    p_spam = sum(y_train) / total
    p_ham = 1 - p_spam
    return p_spam, p_ham

p_spam, p_ham = calculate_prior(y_train)
calculate_prior(y_train)
from collections import defaultdict

def calculate_likelihood(X_train, y_train):
    spam_words = defaultdict(int)
    ham_words = defaultdict(int)
    total_spam, total_ham = 0, 0

    for i, message in enumerate(X_train):
        for word in message:
            if y_train.iloc[i] == 1:  # Spam
                spam_words[word] += 1
                total_spam += 1
            else: 
                ham_words[word] += 1
                total_ham += 1

    return spam_words, ham_words, total_spam, total_ham

spam_words, ham_words, total_spam, total_ham = calculate_likelihood(X_train, y_train)
#calculate_likelihood(X_train, y_train)
def classify(message, spam_words, ham_words, total_spam, total_ham, p_spam, p_ham, alpha=1):
    words = preprocess(message)
    p_spam_given_message = p_spam
    p_ham_given_message = p_ham

    for word in words:
        p_spam_given_message *= (spam_words[word] + alpha) / (total_spam + alpha * len(spam_words))
        p_ham_given_message *= (ham_words[word] + alpha) / (total_ham + alpha * len(ham_words))

    return 1 if p_spam_given_message > p_ham_given_message else 0
y_pred = X_test.apply(lambda x: classify(' '.join(x), spam_words, ham_words, total_spam, total_ham, p_spam, p_ham))


# Calculate accuracy
accuracy = sum(y_pred == y_test) / len(y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')


def predict_and_evaluate(new_message):

    prediction = classify(new_message, spam_words, ham_words, total_spam, total_ham, p_spam, p_ham, alpha=1)
    # Output the prediction
    if prediction == 1:
        result = "The message is spam."
    else:
        result = "The message is not spam."


    return result

# Example usage
new_message = input("Enter the new mail: ")
result = predict_and_evaluate(new_message)
print(result)



























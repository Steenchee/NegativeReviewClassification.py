import pandas as pd
import matplotlib.pyplot as plt
import scipy
from joblib import dump
import plotly.express as px
import numpy as np

dataset = pd.read_csv('CleanedData.csv', sep='\t', encoding='utf-8')
print((dataset.head()))
print(dataset.groupby('class').size())

# распределение отзывов по меткам
# am = dataset['class id'].value_counts()
# am.plot.pie()
# plt.show()
# dataset['class id'].value_counts().plot(kind='bar', xlabel='метка класса', ylabel='частота')
# plt.show()

# Разделение данных
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
def cv(data):
    count_vectorizer = CountVectorizer()
    emb = count_vectorizer.fit_transform(data)
    return emb, count_vectorizer

X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=40, stratify=dataset['class'])

y_train = X_train["class id"].tolist()
X_train = X_train["text_clean_lemma_as_str"].tolist()
y_test = X_test["class id"].tolist()
X_test = X_test["text_clean_lemma_as_str"].tolist()


X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)


# метрики для оценки точности
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
def get_metrics(y_test, y_predicted):
    # true positives / (true positives+false positives)
    precision = precision_score(y_test, y_predicted, pos_label=None, average='weighted')
    # true positives / (true positives + false negatives)
    recall = recall_score(y_test, y_predicted, pos_label=None, average='weighted')

    # harmonic mean of precision and recall
    f1 = f1_score(y_test, y_predicted, pos_label=None, average='weighted')

    # true positives + true negatives/ total
    accuracy = accuracy_score(y_test, y_predicted)
    return accuracy, precision, recall, f1

# логистическия регрессия
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)
y_predicted_counts = clf.predict(X_test_counts)

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

# наивный байес
from sklearn.naive_bayes import MultinomialNB

clf = MultinomialNB()
clf.fit(X_train_counts, y_train)
y_predicted_counts = clf.predict(X_test_counts)

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X_train_counts, y_train)
y_predicted_counts = clf.predict(X_test_counts)

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))


from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
clf.fit(X_train_counts, y_train)
y_predicted_counts = clf.predict(X_test_counts)

accuracy, precision, recall, f1 = get_metrics(y_test, y_predicted_counts)
print("accuracy = %.3f, precision = %.3f, recall = %.3f, f1 = %.3f" % (accuracy, precision, recall, f1))

# Финал

X_train = dataset["text_clean_lemma_as_str"].tolist()
y_train = dataset["class id"].tolist()

X_train_counts, count_vectorizer = cv(X_train)
dump(count_vectorizer, "vectorizer.pkl")
clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg',
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)
dump(clf, 'model.joblib')

# dataset = pd.read_excel('М.Тех_ТЗ_Датасет_DS_NLP.xlsx', index_col=0)
# print(dataset['text'].shape)
# print(dataset['text'].shape[0])
# k = get_result(dataset['text'])
# print(k)
# print(k.shape)
# for i in range(k.shape[0]):
#     print(i)
#     if k[i] != dataset['class'][i]:
#         print('LOX')
#         print(k[i], '\n', dataset['class'][i])
#         break
#     print('KRASAVA')
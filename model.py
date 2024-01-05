import pandas as pd
import spacy
from nltk.corpus import stopwords
from joblib import load


def get_result(text):
    text = text.replace(r'[^\w\s]', ' ', regex=True).replace(r'\s+', ' ', regex=True).str.lower()
    # лемматизация
    nlp = spacy.load("ru_core_news_lg")
    lemma = []
    for doc in nlp.pipe(text.values):
        lemma.append([n.lemma_ for n in doc])
    text = pd.Series(lemma)
    # убираем стоп-слова
    stopwords_ru = stopwords.words("russian")
    text = text.apply(lambda x: [item for item in x if item not in stopwords_ru])
    text = [' '.join(map(str, l)) for l in text]

    # Разделение данных
    count_vectorizer = load('vectorizer.pkl')
    X_test_counts = count_vectorizer.transform(text)
    clf = load('model.joblib')
    list = ['Консультация КЦ', 'Компетентность продавцов/ консультантов',
            'Электронная очередь', 'Доступность персонала в магазине',
            'Вежливость сотрудников магазина', 'Обслуживание на кассе',
            'Обслуживание продавцами/ консультантами',
            'Время ожидания у кассы']
    y_predicted_counts = pd.Series(clf.predict(X_test_counts))
    y_predicted_counts = y_predicted_counts.replace(range(8), list)
    return y_predicted_counts

test = pd.Series(['Мальчики грубые ругают кричат',
                   'Ждала 10 часов у кассы, а кассир-то и не пришел!',
                   'Сдачу не дали, сказали размена не, мдаа ну и кассиры !!!',
                   'Консультанты сами не знают что продают'])

print(test)
print(get_result(test))
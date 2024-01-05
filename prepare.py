# подключение библиотек
import pandas as pd
import spacy
from nltk.corpus import stopwords

dataset = pd.read_excel('М.Тех_ТЗ_Датасет_DS_NLP.xlsx', index_col=0) #загрузка базы данных

# ПРЕДОБРАБОТКА ТЕКСТА ОТЗЫВОВ
# удаление знаков препинания
dataset['text_clean'] = dataset['text'].replace(r'[^\w\s]', ' ', regex=True).replace(r'\s+', ' ', regex=True).str.lower()
print([dataset['class'].unique()])
dataset["class id"] = dataset['class'].replace(dataset['class'].unique(), range(8)) # добавим id меток классов
# лемматизация
nlp = spacy.load("ru_core_news_lg")
lemma = []
for doc in nlp.pipe(dataset["text_clean"].values):
    lemma.append([n.lemma_ for n in doc])
dataset['text_clean_lemma'] = lemma
# убираем стоп-слова
stopwords_ru = stopwords.words("russian")
dataset['text_clean_lemma'] = dataset['text_clean_lemma'].apply(lambda x: [item for item in x if item not in stopwords_ru])
dataset['text_clean_lemma_as_str'] = [' '.join(map(str, l)) for l in dataset['text_clean_lemma']]
dataset.drop(["id", "text", 'text_clean', 'text_clean_lemma'], axis=1, inplace=True) #удаляю лишние колонки
dataset["class id"] = dataset['class'].replace(dataset['class'].unique(), range(8)) # добавим id меток классов

dataset.to_csv('CleanedData.csv', sep='\t', encoding='utf-8') #сохраняю новый файл с подготовленной датой

#print(dataset.head())
#print(dataset[dataset['class'] == 'Консультация КЦ']) #а это просто так


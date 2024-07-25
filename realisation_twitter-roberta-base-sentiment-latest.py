# реализация модели cardiffnlp/twitter-roberta-base-sentiment-latest,
# которая показала себя лучше всех по результатам анализа из файла README

from transformers import AutoModelForSequenceClassification
from transformers import TFAutoModelForSequenceClassification #мы это не используем, но вы можете этот вариант реализации тоже попробовать
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import json

# эта функция не нужна, так как в наших данных уже все нормально
# это нужно для именно твитов, особенно если при парсинге остались ненужные символы в дате
# def preprocess(text):
#     new_text = []
#     for t in text.split(" "):
#         t = '@user' if t.startswith('@') and len(t) > 1 else t
#         t = 'http' if t.startswith('http') else t
#         new_text.append(t)
#     return " ".join(new_text)

# достаем датасет
with open('Arma_3.jsonlines', 'r') as f:
    data = [json.loads(line) for line in f]
reviews = [item['review'] for item in data]
reviews = [review.lower() for review in reviews]

# определяем модель, которую выбрали
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"

# загружаем предобученный токенизатор для этой модели языка из репозитория Hugging Face (слова - цифры)
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# загружаем конфигурацию модели, т.е. набор параметров, которые определяют архитектуру
# и поведение модели.
# это именно:
# архитектура модели (количество слоев, тип слоев, количество attention heads)
# размерность вектора эмбеддинга (embedding size),
# тип оптимизатора и его параметры,
# параметры регуляризации
config = AutoConfig.from_pretrained(MODEL)
# PT

# загружаем предобученную модель языка для задачи классификации последовательностей
# с указанной моделью языка, хранящейся в переменной MODEL.
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

#model.save_pretrained(MODEL) # это закомменчено, потому что у меня нет места на компе, чтобы это сохранить

# text = "Covid is a really bad deasiase!" # пример текста от разрабов
# text = preprocess(text) #обработка, которая не нужна нам (см. выше)
text = reviews[3]
print(reviews[3]) #смотрим, что за отзыв, чтобы прочитать и понять, насколько верно работает модель

# преобразуем текст в формат, пригодный для ввода в модель языка, используя токенизатор. pt - это как раз формат тензоров
encoded_input = tokenizer(text, return_tensors='pt')
# используем модель языка (model) для обработки преобразованного текста (encoded_input) и получаем выходные данные (output) от модели
output = model(**encoded_input)
# извлекаем и преобразуем выходные данные (output) от модели языка в формат, пригодный для дальнейшей обработки.
# то есть преобразуем scores в формат массива NumPy
# detach() отсоединяет тензор от графа вычислений PyTorch.
# это означает, что тензор больше не будет участвовать
# в обратном распространении ошибки и не будет хранить информацию о градиентах.
scores = output[0][0].detach().numpy()
# нормализуем выходные значения, чтобы получить вероятности принадлежности к различным классам
scores = softmax(scores)

# это иной вариант реализации модели, только с использованием формата tf для тенсоров.
# # TF
# model = TFAutoModelForSequenceClassification.from_pretrained(MODEL)
# model.save_pretrained(MODEL)
# text = "Covid cases are increasing fast!"
# encoded_input = tokenizer(text, return_tensors='tf')
# output = model(encoded_input)
# scores = output[0][0].numpy()
# scores = softmax(scores)
# Print labels and scores

# сортруем в порядке возрастания
ranking = np.argsort(scores)
# меняем порядок следования
ranking = ranking[::-1]

for i in range(scores.shape[0]):
    # l = config.id2label[ranking[i]] - берет индекс i из массива ranking
    # и использует его для доступа к словарю config.id2label.
    # Словарь id2label содержит соответствие между индексами и метками (labels).
    # Таким образом, l получает метку, соответствующую индексу ranking[i].
    l = config.id2label[ranking[i]]
    # вытаскиваем вероятность соответствующего индекса
    s = scores[ranking[i]]
    # выводим полученные результаты
    print(f"{i+1}) {l} {np.round(float(s), 4)}")

<h1>ML for games and software</h1>
<h3>This repository is created to share what I've done during BHS Camp.</h3>
<h3>It consists of my homeworks (named with "task..") and final project.</h3>

Bear Head Studio is a 
[young indie studio](https://bearheadstudio.ru/?ysclid=lymslkpk4l937871213).

Note, that while completing hometasks I do not add data, which was used to train these models since the presented code is not my complete project, but is written as an introduction to classical models. For my own training purposes.

<br>
<br>
<br>

<h1>Analysis of various latest and trending Hugging Face architectures that define the mood well in video game reviews</h1>
<h2>
All tests are conducted using the same  

[dataset](https://github.com/mulhod/steam_reviews/blob/master/data/Arma_3.jsonlines)
</h2>

<h3>Model №1

`michellejieli/emotion_text_classifier`

</h3>

![img](https://github.com/user-attachments/assets/dabbfc63-f876-4c3b-b11d-d3aa9d464067)

<b>Theoretical Description of the Model:</b>

DistilRoBERTa-base is a transformer model that performs sentiment analysis. 
I fine-tuned the model on transcripts from the Friends show with the goal of classifying emotions from text data, 
specifically dialogue from Netflix shows or movies. The model predicts 6 Ekman emotions and a neutral class. 
These emotions include anger, disgust, fear, joy, neutrality, sadness, and surprise.

<b>Differences from Traditional Models:</b>

Emotion detection: The model's fine-tuning for emotion classification allows it to learn task-specific knowledge and representations that are not present in traditional language models.

Domain-specific knowledge: The model's training on a large corpus of text data allows it to capture nuances of language and emotion specific to the domain.

Multi-class classification: The model's ability to classify text into multiple emotion categories makes it more effective for emotion detection compared to traditional models.


````python
from transformers import pipeline
import json

with open('Arma_3.jsonlines', 'r') as f:
    data = [json.loads(line) for line in f]
reviews = [item['review'] for item in data]
reviews = [review.lower() for review in reviews]
clf = pipeline(
    task='sentiment-analysis',
    model='michellejieli/emotion_text_classifier',
    truncation=True
)
predictions = clf(reviews[:1000])

fear = 0
neutral = 0
joy = 0
sadness = 0
disgust = 0
surprise = 0
anger = 0

total_predictions = 0
for pred in predictions:
    if pred['label'] == 'fear':
        fear += 1
    elif pred['label'] == 'neutral':
        neutral += 1
    elif pred['label'] == 'joy':
        joy += 1
    elif pred['label'] == 'sadness':
        sadness += 1
    elif pred['label'] == 'disgust':
        disgust += 1
    elif pred['label'] == 'surprise':
        surprise += 1
    elif pred['label'] == 'anger':
        anger += 1
total_predictions = fear + neutral + joy + sadness + disgust + surprise + anger

print("fear reviews:", (fear / total_predictions) * 100)
print("neutral reviews:", (neutral / total_predictions) * 100)
print("joy reviews:", (joy / total_predictions) * 100)
print("sadness reviews:", (sadness / total_predictions) * 100)
print("disgust reviews:", (disgust / total_predictions) * 100)
print("surprise reviews:", (surprise / total_predictions) * 100)
print("anger reviews:", (anger / total_predictions) * 100)
````
The result:
````fear reviews: 1.2
neutral reviews: 38.6
joy reviews: 46.2
sadness reviews: 6.3
disgust reviews: 0.6
surprise reviews: 1.5
anger reviews: 5.6000000000000005
````


![Рисунок7](https://github.com/user-attachments/assets/93a17d53-7c45-4352-8d14-560aee80bbcd)

<h3>Model №2

`distilbert/distilbert-base-uncased-finetuned-sst-2-english`

</h3>

![img_1](https://github.com/user-attachments/assets/1cfb94fc-3ac9-49c9-9dcf-1a48a17a5d75)


<b>Theoretical Description of the Model:</b>
This model is a fine-tune checkpoint of DistilBERT-base-uncased, 
fine-tuned on SST-2. This model reaches an accuracy of 91.3 on the dev set 
(for comparison, Bert bert-base-uncased version reaches an accuracy of 92.7).

📂 The authors use the following Stanford Sentiment Treebank(sst2) corpora for the model.

DistilBERT is a distilled version of the BERT 
(Bidirectional Encoder Representations from Transformers) model, 
which is a transformer-based language model developed by Google. 
DistilBERT is smaller, faster, and more efficient than BERT, 
making it suitable for deployment on devices with limited computational resources.

<b>Key Features:</b>

Smaller size: 
DistilBERT has 66M parameters, compared to BERT's 110M parameters, making it more lightweight and efficient.

Faster inference: DistilBERT is 71% faster than BERT on mobile devices, making it suitable for real-time applications.

Comparable performance: DistilBERT achieves similar performance to BERT on many NLP tasks, including sentiment analysis, question-answering, and text classification.

<b>Differences from Traditional Models:</b>

Transformer architecture: DistilBERT uses a transformer architecture, which is different from traditional recurrent neural networks (RNNs) like GRU and LSTM. Transformers are more parallelizable and can handle longer input sequences.

Multi-layer bidirectional encoding: DistilBERT uses a multi-layer bidirectional encoding scheme, which allows it to capture both local and global context in text data.

Parallelization: DistilBERT's transformer architecture allows for parallelization, making it faster and more efficient than traditional RNNs.

Handling long-range dependencies: DistilBERT's multi-layer bidirectional encoding scheme allows it to capture long-range dependencies in text data, which can be challenging for traditional RNNs.

Pre-training: DistilBERT's pre-training on a large corpus of text data allows it to learn general language representations that can be fine-tuned for specific tasks, making it more effective than traditional models that require task-specific training data.

<b>To sum up,</b> it is an improved version of the BERT model, modified based on the Stanford Sentiment Treebank (SST-2) dataset for sentiment analysis. 
This model provides a good balance between performance and computational efficiency, which makes it suitable for real-time applications.

````python
from transformers import pipeline
import json

with open('Arma_3.jsonlines', 'r') as f:
    data = [json.loads(line) for line in f]
reviews = [item['review'] for item in data]
reviews = [review.lower() for review in reviews]
clf = pipeline(
    task='sentiment-analysis',
    model='distilbert/distilbert-base-uncased-finetuned-sst-2-english',
    truncation=True
)
print(reviews[2])
predictions = clf(reviews[:1000])

positive_reviews = 0
toxic_reviews = 0
total_predictions = 0
for pred in predictions:
    if pred['label'] == 'POSITIVE':
        positive_reviews += 1
    elif pred['label'] == 'NEGATIVE':
        toxic_reviews += 1
total_predictions = positive_reviews + toxic_reviews

print("positive reviews:", (positive_reviews / total_predictions) * 100, "toxic reviews:", (toxic_reviews / total_predictions) * 100)
````

The result:
````
positive reviews: 65.10000000000001 
toxic reviews: 34.9 
````

![Рисунок6](https://github.com/user-attachments/assets/c0a93249-7aa1-4003-85da-eec9e2018a94)


<h3>Model №3

`nlptown/bert-base-multilingual-uncased-sentiment`

</h3>

![img_2](https://github.com/user-attachments/assets/74df1dbf-7561-4464-ab1c-084d5b3b61ee)

<b>Theoretical Description of the Model:</b>

The nlptown/bert-base-multilingual-uncased-sentiment model is a fine-tuned version 
of the BERT (Bidirectional Encoder Representations from Transformers) model, 
specifically designed for sentiment analysis on product reviews in six languages:
English, Dutch, German, French, Spanish, and Italian. 
This model predicts the sentiment of a review as a number of stars (between 1 and 5).

<b>Key Features:</b>

Multilingual support:
The model is trained on a multilingual dataset, allowing it to perform sentiment analysis on reviews in six languages.

Fine-tuned for sentiment analysis: The model is fine-tuned on a large dataset of product reviews, making it specifically suited for sentiment analysis tasks.

High accuracy: The model has achieved high accuracy on a held-out dataset of 5,000 product reviews in each language.

<b>Differences from Traditional Models:</b>

Transformer architecture: 
The model uses a transformer architecture, which is different from traditional recurrent neural networks (RNNs) like GRU and LSTM. Transformers are more parallelizable and can handle longer input sequences.

Pre-training: The model is pre-trained on a large corpus of text data, which allows it to learn general language representations that can be fine-tuned for specific tasks.

Multilingual support: The model's multilingual support sets it apart from traditional models, which are often limited to a single language.


Handling long-range dependencies: 
The model's transformer architecture allows it to capture long-range dependencies in text data, which can be challenging for traditional RNNs.

<b>In summary,</b> the nlptown/bert-base-multilingual-uncased-sentiment model is a powerful tool for sentiment analysis on product reviews in multiple languages, offering high accuracy and efficiency compared to traditional models.

````python
from transformers import pipeline
import json

with open('Arma_3.jsonlines', 'r') as f:
    data = [json.loads(line) for line in f]
reviews = [item['review'] for item in data]
reviews = [review.lower() for review in reviews]
clf = pipeline(
    task='sentiment-analysis',
    model='nlptown/bert-base-multilingual-uncased-sentiment',
    truncation=True
)

predictions = clf(reviews[:1000])

five_stars = 0
four_stars = 0
three_stars = 0
two_stars = 0
one_star = 0
total_predictions = 0
for pred in predictions:
    if pred['label'] == '5 stars':
        five_stars += 1
    elif pred['label'] == '4 stars':
        four_stars += 1
    elif pred['label'] == '3 stars':
        three_stars += 1
    elif pred['label'] == '2 stars':
        two_stars += 1
    elif pred['label'] == '1 star':
        one_star += 1
total_predictions = five_stars + four_stars + three_stars + two_stars + one_star

print("5 stars:", (five_stars / total_predictions) * 100)
print("four stars:", (four_stars / total_predictions) * 100)
print("three stars:", (three_stars / total_predictions) * 100)
print("two stars:", (two_stars / total_predictions) * 100)
print("one star:", (one_star / total_predictions) * 100)
````

The result:
````
five stars: 44.0
four stars: 26.1
three stars: 7.3999999999999995
two stars: 6.9
one star: 15.6
````
![Рисунок5](https://github.com/user-attachments/assets/39822ff9-0233-484b-9e03-83e82fcd3b27)

<h3>Model №4

`ProsusAI/finbert`

</h3>

![img_3](https://github.com/user-attachments/assets/b3a806f6-b669-40ae-8ff5-bea106a5233d)


💜️🤗 Interesting fact: this model is the first in the list of those with the highest number of stars on Hugging Face!

<b>Theoretical Description of the Model:</b>
FinBERT is a pre-trained NLP model to analyze sentiment of financial text. 
It is built by further training the BERT language model in the finance domain, 
using a large financial corpus and thereby fine-tuning it for financial sentiment classification. 
Financial PhraseBank by Malo et al. (2014) is used for fine-tuning. 
For more details, please see the paper FinBERT: 
Financial Sentiment Analysis with Pre-trained Language Models and our related blog post on Medium.
The model will give softmax outputs for three labels: positive, negative or neutral.

❗️The model has a problem that it can recognize, for example, such a review:

"downloaded arma, and then opened it. proceeded to do a horrible tutorial, but after that, it was just beautiful. opened an rpg server, and ran 2 miles to get to an atm, and ran 2 miles to a shady car salesman. bought a car, spent the next half hour trying to figure out how to actually get into it. after i finally managed to glitch into it, i underestimated it's ability to travel at light speed, which in turn caused me to crash into a wall. after that, my car was a smoking mess, with a top speed that a sloth could outdo. 10/10 would crash again"

as negative because of the words "horrible tutorial", etc. But in fact, this is a good review, because the caller writes: "10/10 would crash again".
But apparently this part of the recall does not fit into the data size for the model, the parameter `truncation=True`, so it loses key data.

✅ But I am glad that, for example, the review:

"good, realistic.... f**k it, its arma 3-the best game ever 11/10"

the model recognizes as positive, though with a not very high score: `'score': 0.6725947260856628`, most likely because of the "f**k" word.

<b>Key Features:</b>

Domain-specific pre-training: 
The model is pre-trained on a large corpus of financial text data, allowing it to learn domain-specific knowledge and representations.

Financial terminology understanding: The model is trained to understand financial terminology, including industry-specific jargon and concepts.

Contextualized embeddings: The model uses contextualized embeddings, which capture the nuances of language and allow for more accurate sentiment analysis and entity recognition.

Fine-tuning for specific tasks: The model can be fine-tuned for specific financial NLP tasks, such as sentiment analysis, entity recognition, and question-answering.

<b>Differences from Traditional Models:</b>

Handling complex financial text: The model's ability to handle long-range dependencies and capture complex relationships in financial text data makes it more effective than traditional models.

Contextualized embeddings: The model's use of contextualized embeddings allows it to capture the nuances of language, which is not possible with traditional word embeddings.

Transformer architecture: The model's transformer architecture allows it to handle long-range dependencies and capture complex relationships in financial text data.

Pre-training and fine-tuning: FinBERT's pre-training on a large corpus and fine-tuning on a specific task (sentiment analysis) allows it to learn general language representations and adapt to the specific task at hand.

Bi-directionality: FinBERT's bi-directional architecture allows it to capture both forward and backward dependencies in text data, which is not possible with traditional RNNs.

<b>Overall,</b> FinBERT's combination of domain-specific knowledge, contextualized embeddings, transformer architecture, pre-training, and fine-tuning, and bi-directionality make it a more effective model for financial sentiment analysis than classical RNN, BERT, and Siamese networks.


````python
from transformers import pipeline
import json

with open('Arma_3.jsonlines', 'r') as f:
    data = [json.loads(line) for line in f]
reviews = [item['review'] for item in data]
reviews = [review.lower() for review in reviews]
clf = pipeline(
    task='sentiment-analysis',
    model='ProsusAI/finbert',
    truncation=True
)

print(reviews[95])
predictions = clf(reviews[:1000])
# print(predictions)


negative = 0
neutral = 0
positive = 0
total_predictions = 0
for pred in predictions:
    if pred['label'] == 'negative':
        negative += 1
        print("neg")
        print(pred)
    elif pred['label'] == 'neutral':
        neutral += 1
        print("neu")
        print(pred)
    elif pred['label'] == 'positive':
        positive += 1
        print("pos")
        print(pred)

total_predictions = negative + positive + neutral

print("negative:", (negative / total_predictions) * 100)
print("neutral:", (neutral / total_predictions) * 100)
print("positive:", (positive / total_predictions) * 100)
````

The result:
````
positive: 5.6000000000000005
neutral: 89.60000000000001
negative: 4.8
````

![Рисунок4](https://github.com/user-attachments/assets/b5d92ace-b9c7-430c-8d51-5cc0e2368b34)



<h3>Model №5

`cardiffnlp/twitter-roberta-base-sentiment`

</h3>

![img_4](https://github.com/user-attachments/assets/41a5bbc5-7fef-43ef-8713-e05a8e67cf53)

<b>Theoretical Description of the Model:</b>
This is a RoBERTa-base model trained on ~124M tweets from January 2018 to December 2021, 
and finetuned for sentiment analysis with the TweetEval benchmark. 
The original Twitter-based RoBERTa model can be found here and the original
reference paper is TweetEval. This model is suitable for English.

The cardiffnlp/twitter-roberta-base-sentiment model is a RoBERTa-based model 
specifically designed for sentiment analysis on Twitter data. 
It is trained on a large corpus of ~124M tweets and fine-tuned for sentiment analysis with the TweetEval benchmark.


<b>Key Features:</b>

Domain-specific knowledge: The model is trained on a large corpus of Twitter data, allowing it to learn domain-specific knowledge and representations.

RoBERTa architecture: The model uses the RoBERTa architecture, which is a variant of BERT that is trained with a masked language modeling objective and a next sentence prediction objective.

Fine-tuning for sentiment analysis: The model is fine-tuned for sentiment analysis with the TweetEval benchmark, allowing it to learn task-specific knowledge and representations.

Three-class sentiment analysis: The model is trained to predict three classes of sentiment: positive, negative, and neutral.

<b>Differences from Traditional Models:</b>

Twitter-specific sentiment analysis: The model's domain-specific knowledge and training on Twitter data make it more effective for sentiment analysis on Twitter data compared to traditional models.

Improved performance on out-of-distribution data: The model's fine-tuning for sentiment analysis and use of the RoBERTa architecture allow it to perform better on out-of-distribution data compared to traditional models.

More accurate sentiment analysis: The model's three-class sentiment analysis and fine-tuning for sentiment analysis allow it to perform more accurate sentiment analysis compared to traditional models.


````python
from transformers import pipeline
import json

with open('Arma_3.jsonlines', 'r') as f:
    data = [json.loads(line) for line in f]
reviews = [item['review'] for item in data]
reviews = [review.lower() for review in reviews]
clf = pipeline(
    task='sentiment-analysis',
    model='cardiffnlp/twitter-roberta-base-sentiment',
    max_length = 510,
    truncation=True
)

predictions = clf(reviews[:1000])
negative = 0
neutral = 0
positive = 0
total_predictions = 0
for pred in predictions:
    if pred['label'] == 'LABEL_0':
        negative += 1
    elif pred['label'] == 'LABEL_1':
        neutral += 1
    elif pred['label'] == 'LABEL_2':
        positive += 1

total_predictions = negative + positive + neutral

print("negative:", (negative / total_predictions) * 100)
print("neutral:", (neutral / total_predictions) * 100)
print("positive:", (positive / total_predictions) * 100)

```` 
The result:
````
negative: 22.3
neutral: 14.7
positive: 63.0
````

![Рисунок3](https://github.com/user-attachments/assets/cbfd3ebb-e71e-488f-aff9-aa7577e3343e)


<h3>Model №6

`cardiffnlp/twitter-roberta-base-sentiment-latest`

</h3>

![img_5](https://github.com/user-attachments/assets/60ca8d01-62f0-4cbb-9f3c-056987bc1260)

This is a RoBERTa-base model trained on ~124M tweets from January 2018 to December 2021, 
and finetuned for sentiment analysis with the TweetEval benchmark. 
The original Twitter-based RoBERTa model can be found here and the original 
reference paper is TweetEval. This model is suitable for English.

<b>Theoretical Description of the Model:</b>

The cardiffnlp/twitter-roberta-base-sentiment-latest model is a sentiment analysis model 
specifically designed for Twitter data. It is a RoBERTa-base model trained on approximately 
124 million tweets from January 2018 to December 2021 and fine-tuned for sentiment analysis
with the TweetEval benchmark.

<b>Key Features:</b>

Twitter-specific training data: The model is trained on a large corpus of Twitter data, allowing it to learn domain-specific knowledge and representations.

Sentiment analysis: The model is fine-tuned for sentiment analysis, allowing it to learn task-specific knowledge and representations.

RoBERTa-base architecture: The model uses the RoBERTa-base architecture, which is a variant of the BERT architecture that is specifically designed for sentiment analysis tasks.

<b>Differences from Traditional Models:</b>

Improved performance on out-of-distribution data: The model's training on a large corpus of Twitter data and fine-tuning for sentiment analysis allow it to perform better on out-of-distribution data compared to traditional models.

More accurate sentiment classification: The model's RoBERTa-base architecture and task-specific fine-tuning allow it to perform more accurate sentiment classification compared to traditional models.


````python
from transformers import pipeline
import json

with open('Arma_3.jsonlines', 'r') as f:
    data = [json.loads(line) for line in f]
reviews = [item['review'] for item in data]
reviews = [review.lower() for review in reviews]
clf = pipeline(
    task='sentiment-analysis',
    model='cardiffnlp/twitter-roberta-base-sentiment-latest',
    truncation=True,
    max_length=510,
)
predictions = clf(reviews[:1000])

negative = 0
neutral = 0
positive = 0


total_predictions = 0
for pred in predictions:
    if pred['label'] == 'negative':
        negative += 1
    elif pred['label'] == 'neutral':
        neutral += 1
    elif pred['label'] == 'positive':
        positive += 1
total_predictions = negative + neutral + positive

print("negative reviews:", (negative / total_predictions) * 100)
print("neutral reviews:", (neutral / total_predictions) * 100)
print("positive reviews:", (positive / total_predictions) * 100)
````

The result:

````commandline
positive reviews: 67.5
neutral reviews: 9.700000000000001
negative reviews: 22.8
````

![Рисунок1](https://github.com/user-attachments/assets/4fddb5cc-7075-49b0-8605-f3ae28063809)


<h3>And now let's compare the results of processing this dataset on different 
models and look at the similarities and differences</h3>


| Model 1           | Model 2           | Model 3           |
|-------------------|-------------------|-------------------|
| ![Рисунок7](https://github.com/user-attachments/assets/0e2b90da-178d-44b0-9408-167b093c092d)| ![Рисунок6](https://github.com/user-attachments/assets/8573e6dc-3376-4b87-93ea-68fe56cc689c) |![Рисунок5](https://github.com/user-attachments/assets/4c000716-4425-462c-be24-9ad321a740aa)|
| Model 4           | Model 5           | Model 6           |
| ![Рисунок4](https://github.com/user-attachments/assets/ad7e7f04-23c9-44fb-9885-14267a51af4c)|![Рисунок3](https://github.com/user-attachments/assets/dfc1403d-7f54-4860-9d73-4c9fc3a2ea9d)|![Рисунок1](https://github.com/user-attachments/assets/1b925b2e-e649-4aad-a735-eb16c516fc64)|


📝 Analyzing all the diagrams obtained, we can conclude that models with numbers 2️⃣, 3️⃣, 5️⃣, 6️⃣ cope best with this task. 
Model 4️⃣ coped with the task the worst (most likely due to the fact that it was trained on financial data, 
and in the dataset used a completely different vocabulary (game topics) and the style of speech is conversational, 
not formal or businesslike. The model 1️⃣ also performed poorly, although, unlike the model 4️⃣, it has no excuses, 
because it was fed conversation data from Netflix movies and shows, which in terms of speech style is very suitable for the dataset used. 
So we can also conclude that the number of downloads of the model can also illustrate its performance, and yes, model 1️⃣ has the smallest one.

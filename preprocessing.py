import csv
import string
import pandas as pd
import re as RE
from nltk.corpus import stopwords as SW
from nltk.stem.porter import PorterStemmer as PS
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud


def preprocessing():
    reviews = pd.read_csv('reviews.csv')

    # stopwords filtering
    # nltk.download('stopwords')
    stop = set(SW.words('english'))
    punctuation = list(string.punctuation)
    stop.update(punctuation)

    # Initialize empty array for appending clean text
    corpus = []
    filtered_review = []

    for review1 in reviews['review']:
        review1 = RE.sub('[^a-zA-Z]', ' ', review1)
        review1 = review1.lower()
        review1 = review1.split()

        # create PorterStemmer object to take main stem of each word
        ps1 = PS()
        review1 = [ps1.stem(word) for word in review1 if not word in stop]
        review1_corpus = []
        for j in review1:
            corpus.append(j.strip())
            review1_corpus.append(j.strip())

        filtered_review.append(' '.join(review1_corpus))

    reviews['filtered_review'] = filtered_review

    reviews.to_csv("reviews_filtered.csv", sep=',', index=False)

    # Most Common words
    counter = Counter(corpus)
    most_common = counter.most_common(10)
    most_common = dict(most_common)
    print(most_common)
    # {'movi': 212,
    # 'film': 189,
    # 'like': 116,
    # 'one': 101,
    # 'charact': 101,
    # 'end': 64,
    # 'first': 63,
    # 'stori': 62,
    # 'get': 62,
    # 'time': 61}

    # review text
    # plt.figure(figsize=(20, 20))
    # wc = WordCloud(max_words=2000, width=1600, height=800).generate(" ".join(reviews[reviews.star >= 6].review))
    # plt.imshow(wc, interpolation='bilinear')
    # plt.savefig('good_movie_review.jpg')
    #
    # plt.figure(figsize=(20, 20))
    # wc = WordCloud(max_words=2000, width=1600, height=800).generate(" ".join(reviews[reviews.star < 6].review))
    # plt.imshow(wc, interpolation='bilinear')
    # plt.savefig('bad_movie_review.jpg')


if __name__ == '__main__':
    preprocessing()
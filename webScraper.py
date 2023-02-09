import csv
import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen


def webscraper(title_id, movie):
    url = 'https://www.imdb.com/title/'+title_id+'/reviews?ref_=tt_urv'
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    main_page = soup.find(id='main')
    content_list = main_page.find_all("div", class_='review-container')

    review_df = pd.DataFrame(columns=['movie', 'title', 'star', 'review'])

    for content in content_list:
        review_title = content.find("a", class_='title')
        star = content.find("span", class_='rating-other-user-rating')
        if star is None:
            continue
        star = star.text.strip()[:-3]
        review = content.find("div", class_='text show-more__control')
        if review is None:
            review = content.find("div", class_='text show-more__control clickable')

        res = pd.DataFrame([[movie, review_title.text.strip(), star, review.text.strip()]], columns=['movie', 'title', 'star', 'review'])
        review_df = pd.concat([review_df, res], ignore_index=True)

    return review_df


def scraper_for_all():
    total_review = pd.DataFrame(columns=['movie', 'title', 'star', 'review'])
    with open('movie_id.csv') as csv_file:
        movies = csv.reader(csv_file, delimiter=',')
        next(movies, None)
        for row in movies:
            sub_review = webscraper(row[0], row[1])
            total_review = pd.concat([total_review, sub_review], ignore_index=True)

    total_review.to_csv("reviews.csv", sep=',')


if __name__ == '__main__':
    scraper_for_all()

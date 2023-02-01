import pandas as pd
from bs4 import BeautifulSoup
from urllib.request import urlopen


def webscraper():
    url = 'https://www.imdb.com/title/tt1630029/reviews?ref_=tt_urv'
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    main_page = soup.find(id='main')
    content_list = main_page.find_all("div", class_='review-container')

    review_df = pd.DataFrame(columns=['title', 'star', 'review'])

    for content in content_list:
        title = content.find("a", class_='title').text.strip()
        star = content.find("span", class_='rating-other-user-rating').text.strip()
        review = content.find("div", class_='text show-more__control').text.strip()
        if review is None:
            review = content.find("div", class_='text show-more__control clickable').text.strip()

        res = pd.DataFrame([[title, star, review]], columns=['title', 'star', 'review'])
        review_df = pd.concat([review_df, res], ignore_index=True)

    review_df.to_csv("reviews.csv", sep='\t')


if __name__ == '__main__':
    webscraper()

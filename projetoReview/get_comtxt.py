# coding: utf-8

import requests
from bs4 import BeautifulSoup
import csv
import webbrowser
import io


def display(content, filename='output.html'):
    with open(filename, 'wb') as f:
        f.write(content)
        webbrowser.open(filename)


def get_soup(session, url, show=False):
    r = session.get(url)
    if show:
        display(r.content, 'temp.html')

    if r.status_code != 200:  # not OK
        print('[get_soup] status code:', r.status_code)
    else:
        return BeautifulSoup(r.text, 'html.parser')


def post_soup(session, url, params, show=False):
    '''Read HTML from server and convert to Soup'''

    r = session.post(url, data=params)

    if show:
        display(r.content, 'temp.html')

    if r.status_code != 200:  # not OK
        print('[post_soup] status code:', r.status_code)
    else:
        return BeautifulSoup(r.text, 'html.parser')


def scrape(url, lang='ALL'):
    # create session to keep all cookies (etc.) between requests
    session = requests.Session()
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36',
    })

    items = parse(session, url + '?filterLang=' + lang)

    return items


def parse(session, url):
    '''Get number of reviews and start getting subpages with reviews'''

    print('[parse] url:', url)

    soup = get_soup(session, url)

    if not soup:
        print('[parse] no soup:', url)
        return

    num_reviews = soup.find('span', class_='reviews_header_count').text  # get text
    num_reviews = num_reviews[1:-1]
    num_reviews = num_reviews.replace(',', '')
    num_reviews = float(num_reviews)  # convert text into integer
    print('[parse] num_reviews ALL:', num_reviews)

    url_template = url.replace('.html', '-or{}.html')
    print('[parse] url_template:', url_template)

    items = []

    offset = 10

    while (True):
        subpage_url = url_template.format(offset)

        subpage_items = parse_reviews(session, subpage_url)
        if not subpage_items:
            break

        items += subpage_items

        if len(subpage_items) < 5:
            break

        offset += 5

    return items


def get_reviews_ids(soup):
    items = soup.find_all('div', attrs={'data-reviewid': True})

    if items:
        reviews_ids = [x.attrs['data-reviewid'] for x in items][::2]
        print('[get_reviews_ids] data-reviewid:', reviews_ids)
        return reviews_ids


def get_more(session, reviews_ids):
    url = 'https://www.tripadvisor.com/OverlayWidgetAjax?Mode=EXPANDED_HOTEL_REVIEWS_RESP&metaReferer=Hotel_Review'

    payload = {
        'reviews': ','.join(reviews_ids),  
        'widgetChoice': 'EXPANDED_HOTEL_REVIEW_HSX',  
        'haveJses': 'earlyRequireDefine,amdearly,global_error,long_lived_global,apg-Hotel_Review,apg-Hotel_Review-in,bootstrap,desktop-rooms-guests-dust-en_US,responsive-calendar-templates-dust-en_US,taevents',
        'haveCsses': 'apg-Hotel_Review-in',
        'Action': 'install',
    }
    soup = post_soup(session, url, payload)
    return soup


def parse_reviews(session, url):
    '''Get all reviews from one page'''

    print('[parse_reviews] url:', url)

    soup = get_soup(session, url)

    if not soup:
        print('[parse_reviews] no soup:', url)
        return

    #hotel_name = soup.find('h1', id='HEADING').text

    reviews_ids = get_reviews_ids(soup)
    if not reviews_ids:
        return

    soup = get_more(session, reviews_ids)

    if not soup:
        print('[parse_reviews] no soup:', url)
        return

    items = []

    for idx, review in enumerate(soup.find_all('div', class_='reviewSelector')):

        '''
        badgets = review.find_all('span', class_='badgetext')

        if len(badgets) > 0:
            contributions = badgets[0].text
        else:
            contributions = '0'

        if len(badgets) > 1:
            helpful_vote = badgets[1].text
        else:
            helpful_vote = '0'
        user_loc = review.select_one('div.userLoc strong')
        if user_loc:
            user_loc = user_loc.text
        else:
            user_loc = ''
        '''
        bubble_rating = review.select_one('span.ui_bubble_rating')['class']
        bubble_rating = bubble_rating[1].split('_')[-1]

        item = {
            'review_rate': bubble_rating,
            'review_body': review.find('p', class_='partial_entry').text,
            #'review_date': review.find('span', class_='ratingDate')['title'],  # 'ratingDate' instead of 'relativeDate'
        }

        items.append(item)
        print('\n--- review ---\n')
        for key, val in item.items():
            print(' ', key, ':', val)

    print()

    return items


def write_in_csv(items, filename='results.csv',
                 headers=['hotel name', 'review title', 'review body',
                          'review date', 'contributions', 'helpful vote',
                          'user name', 'user location', 'rating'],
                 mode='w'):
    print('--- CSV ---')

    with io.open(filename, mode, encoding="utf-8") as csvfile:
        csv_file = csv.DictWriter(csvfile, headers)

        if mode == 'w':
            csv_file.writeheader()

        csv_file.writerows(items)

def read(file):
    f = open(file, "r")
    for link in f :
        if str(link) not in start_urls:
            start_urls.append(link)
        else:
            pass

DB_COLUMN = 'review_rate'
DB_COLUMN1 = 'review_body'
#DB_COLUMN2 = 'review_date'

start_urls = []
read('restaurantes_get.txt')

lang = 'pt-br'

headers = [
    DB_COLUMN,
    DB_COLUMN1,
    #DB_COLUMN2,
]

for url in start_urls:

    # get all reviews for 'url' and 'lang'
    items = scrape(url, lang)

    if not items:
        print('No reviews')
    else:
        # write in CSV
        filename = url.split('Reviews-')[1][:-5] + '__' + lang
        print('filename:', filename)
        write_in_csv(items, filename + '.csv', headers, mode='w')

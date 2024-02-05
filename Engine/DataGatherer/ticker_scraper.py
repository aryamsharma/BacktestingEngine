import requests
from bs4 import BeautifulSoup
from string import ascii_uppercase

def update_quotes(filepath, update=True):
    if not update:
        return [line.split(",")[0] for line in open(filepath, "r+").readlines()]

    all_quotes = {}

    for letter in ascii_uppercase:
        print(f"Starting {letter}", end ="\r")
        all_quotes.update(scrape_quotes(letter))

    return write_quotes(filepath, all_quotes)

def scrape_quotes(letter):
    quotes = {}
    rows = get_rows(
        f"https://eoddata.com/stocklist/NYSE/{letter}.htm")

    for row in rows:
        columns = [element.text.strip() for element in row.find_all('td')[:2]]
        if not columns:
            continue

        # Looks ugly but just removes the .A shares and stuff like that
        quote = columns[0].split(".")[0].split("-")[0]

        if quotes.get(quote, None) is None:
            quotes[quote] = columns[1]

    return quotes

def get_rows(link):
    soup = BeautifulSoup(requests.get(link).text, features="lxml")
    return soup.find('table', attrs={'class': "quotes"}).find_all('tr')

def write_quotes(filename, quotes):
    written_quotes = []
    with open(filename, "w+") as f:
        for quote in get_sorted_pairs(quotes.items()):
            written_quotes.append(quote[0])
            f.write(",".join(quote) + "\n")

    return written_quotes

def get_sorted_pairs(items):
    return sorted(list(items), key=lambda col: col[0], reverse=False)

def read_quotes(filename):
    with open(filename) as f:
        tickers = [line.split(",")[0] for line in f.readlines()]
    return tickers

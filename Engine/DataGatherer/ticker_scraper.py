import requests
from bs4 import BeautifulSoup
from string import ascii_uppercase
from itertools import chain

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

#

def print_dates(dates):
    for date, tickers in dates.items():
        print(f"{date}: {tickers}")

def remove_downloaded_files(current_files, new):
    converted_current_files = {}

    for ticker_file in current_files:
        date = ticker_file[-14:-4]
        ticker = ticker_file.split("_")[0]

        if converted_current_files.get(date, None) is None:
            converted_current_files[date] = [ticker]
            continue
        converted_current_files[date] += [ticker]

    current_files_list = [
        [(date, ticker) for ticker in tickers] for date, tickers in converted_current_files.items()
    ]
    new_files_list = [
        [(date, ticker) for ticker in tickers] for date, tickers in new.items()
    ]

    current_files_list = list(chain.from_iterable(current_files_list))
    new_files_list = list(chain.from_iterable(new_files_list))

    file_diff = [
        pair for pair in new_files_list if pair not in current_files_list
    ]

    final = {}

    for date, ticker in file_diff:
        if final.get(date, None) is None:
            final[date] = [ticker]
            continue

        final[date] += [ticker]

    return final

def parse_dates(all_dates, ticker_dates, ticker_file):
    for date in ticker_dates:
        if date is None:
            continue

        if all_dates.get(date, None) is None:
            all_dates[date] = [ticker_file[:-4]]
            continue
        all_dates[date] += [ticker_file[:-4]]

    return all_dates

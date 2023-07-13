import requests
import datetime
import os
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json
from sklearn.preprocessing import OneHotEncoder
from transformers import pipeline, AutoTokenizer, AutoModel
import re
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch
import nltk
from keras_preprocessing.sequence import pad_sequences
from GoogleNews import GoogleNews
import finnhub
import xmltodict

# Define the categories
categories = ['Sunny', 'Partly cloudy', 'Cloudy', 'Overcast', 'Mist', 'Patchy rain possible',
              'Patchy snow possible', 'Patchy sleet possible', 'Patchy freezing drizzle possible',
              'Thundery outbreaks possible', 'Blowing snow', 'Blizzard', 'Fog', 'Freezing fog',
              'Patchy light drizzle', 'Light drizzle', 'Freezing drizzle', 'Heavy freezing drizzle',
              'Patchy light rain', 'Light rain', 'Moderate rain at times', 'Moderate rain',
              'Heavy rain at times', 'Heavy rain', 'Light freezing rain',
              'Moderate or heavy freezing rain', 'Light sleet', 'Moderate or heavy sleet',
              'Patchy light snow', 'Light snow', 'Patchy moderate snow', 'Moderate snow',
              'Patchy heavy snow', 'Heavy snow', 'Ice pellets', 'Light rain shower',
              'Moderate or heavy rain shower', 'Torrential rain shower', 'Light sleet showers',
              'Moderate or heavy sleet showers', 'Light snow showers', 'Moderate or heavy snow showers',
              'Light showers of ice pellets', 'Moderate or heavy showers of ice pellets',
              'Patchy light rain with thunder', 'Moderate or heavy rain with thunder',
              'Patchy light snow with thunder', 'Moderate or heavy snow with thunder']


def download_historical_whether(api_key, query, from_date, to_date):
    # Subtract the timedelta object from the from_date to get the to_date
    if to_date is None:
        to_date = datetime.now()
    else:
        to_date = datetime.strptime(to_date, "%Y-%m-%d")
    results = []
    folder_path = "./NLP/whether_data"
    filename = query + "_whether_data.csv"
    # Loop over the dates between from_date and to_date
    dt = datetime.strptime(from_date, "%Y-%m-%d")
    while dt < to_date:
        # Convert the date to a string in the format expected by the API
        dt_str = dt.strftime('%Y-%m-%d')

        # Construct the URL for the API call
        url = f"http://api.weatherapi.com/v1/history.json?key={api_key}&q={query}&dt={dt_str}"

        # Make the API call
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            results.extend(data['forecast']['forecastday'])

        dt = dt + timedelta(days=1)

    # Check if file with the given name exists in the folder
    file_path = os.path.join(folder_path, filename)
    df = pd.DataFrame.from_records(results)

    # Convert the `datetime` column to a pandas datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Set the `datetime` column as the index
    df.set_index('date', inplace=True)

    # Export the DataFrame to a CSV file
    df.to_csv(file_path)


def prepare_time_series_whether_data(data, window_size, output_step, dilation, stride=1):
    features = data.shape[-1]
    n_samples = (len(data) - dilation * (window_size - 1) - output_step)
    X = np.zeros((n_samples, window_size, features))
    features = len(data)

    for i in range(n_samples):
        for j in range(window_size):
            X[i][j] = (data[i + (j * dilation)])
    return X


def prepare_time_series_news_data(data, window_size, output_step, dilation, stride=1):
    features = data.shape[-1]
    n_samples = (len(data) - dilation * (window_size - 1) - output_step)
    X = np.zeros((n_samples, window_size, features))
    features = len(data)

    for i in range(n_samples):
        for j in range(window_size):
            X[i][j] = (data[i + (j * dilation)])
    return X


def prepare_time_series_news_raw_data(data, window_size, output_step, dilation, stride=1):
    features = data.shape[-1]
    n_samples = (len(data) - dilation * (window_size - 1) - output_step)
    X = np.empty((n_samples, window_size), dtype=object)
    features = len(data)

    for i in range(n_samples):
        for j in range(window_size):
            X[i][j] = (data[i + (j * dilation)])
    return X


def prepare_whether_data(stock_df, window_size, from_date, to_date, output_step, new_data=False):
    # Set the API key
    api_key = 'b7a439cb870a4a09be9114748230705'
    # Create the encoder
    encoder = OneHotEncoder(categories=[categories])

    # Set the search parameters
    query1 = "Ha Noi"
    query2 = "Ho Chi Minh"
    query3 = "Da Nang"
    # Define the from_date as the current date and time
    # if new_data:
    #     download_historical_whether(api_key, query1, from_date, to_date)
    #     download_historical_whether(api_key, query2, from_date, to_date)
    #     download_historical_whether(api_key, query3, from_date, to_date)

    # Create an empty list to store day arrays
    day_arrays = []

    # Iterate over all CSV files in the folder
    folder_path = './NLP/whether_data/'
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            # Load CSV file into DataFrame
            df = load_data_with_index(os.path.join(folder_path, filename), stock_df.index)
            # Extract 'day' column into NumPy array and append to list
            dict_column = df['day'].values
            results = []
            for j in dict_column:
                dict = json.loads(j.replace("'", "\""))
                whether = np.array([dict.pop('condition')["text"]])
                encoded_data = np.array(encoder.fit_transform(whether.reshape(-1, 1)).toarray())
                encoded_data = encoded_data.reshape(-1)
                my_array = np.array(list(dict.values()))
                my_array = np.insert(my_array, -1, encoded_data, axis=0)
                results.append(my_array)

            day_arrays.append(results)

    # Merge all day arrays into one NumPy array
    whether_day_array = np.array(day_arrays)
    whether_day_array = whether_day_array.reshape(whether_day_array.shape[1],
                                                  whether_day_array.shape[0] * whether_day_array.shape[2])

    output = prepare_time_series_whether_data(whether_day_array, window_size, output_step, 1)
    return output


def filter_news_by_stock_dates(news_df, stock_df, max_rows):
    # Convert 'date' column to datetime data type
    news_df['date'] = pd.to_datetime(news_df['date'])

    # Count the number of rows for each date in the news dataframe
    news_counts = news_df['date'].value_counts()
    max_days_skipped = 365  # Set a maximum number of days to skip

    # Create a list to store the filtered rows
    filtered_rows = []

    # Iterate over the stock index dates and filter the news dataframe
    for date in stock_df.index:
        # Check if the date is present in the news dataframe
        if date in news_counts:
            # Retrieve the rows for the current date
            current_rows = news_df[news_df['date'] == date]
            # Add the required number of rows to the filtered_rows list
            filtered_rows.extend(current_rows.values.tolist())
            if len(current_rows) < max_rows:
                # Distribute remaining rows evenly to the subsequent dates
                rows_remaining = max_rows - news_counts[date]
                days_skipped = 1
                previous_date = date
                while news_counts[date] < rows_remaining and previous_date - pd.DateOffset(days=days_skipped) in news_df["date"]:
                    previous_date  -= pd.DateOffset(days=days_skipped)
                    rows = news_df[news_df['date'] == previous_date]
                    if len(rows) < 0:
                        days_skipped += 1
                    # Check if the next date has enough rows
                    if news_counts[previous_date] > 0:
                        rows_to_include = news_counts[previous_date] - max_rows
                        if rows[:rows_to_include].values.tolist() not in filtered_rows:
                            filtered_rows.extend(rows[:rows_to_include].values.tolist())
                            news_counts[previous_date] -= rows_to_include
                            news_counts[date] += rows_to_include
                            rows_remaining -= rows_to_include

                    days_skipped += 1
        else:
            # Calculate the number of rows to include for the current date
            rows_remaining = max_rows
            news_counts[date] = 0
            # Distribute remaining rows evenly to the subsequent dates

            days_skipped = 1
            previous_date = date
            while news_counts[date] < rows_remaining: #and previous_date - pd.DateOffset(days=days_skipped) in news_df["date"]:
                previous_date -= pd.DateOffset(days=days_skipped)
                rows = news_df[news_df['date'] == previous_date]
                if len(rows) == 0:
                    days_skipped += 1
                # Check if the next date has enough rows
                elif news_counts[previous_date] > 5:
                        rows_to_include = news_counts[previous_date] - max_rows
                        if rows[:rows_to_include].values.tolist() not in filtered_rows:
                            rows = rows.copy()
                            rows['date'] = date
                            filtered_rows.extend(rows[:rows_to_include].values.tolist())
                            news_counts[previous_date] -= rows_to_include
                            news_counts[date] += rows_to_include
                            rows_remaining -= rows_to_include

                        days_skipped += 1

    # Create a dataframe from the filtered_rows list
    filtered_news_df = pd.DataFrame(filtered_rows, columns=news_df.columns)

    return filtered_news_df


def prepare_news_data(stock_df, symbol, window_size, from_date, to_date, output_step, topK, new_data=False):
    # Read the csv file
    # Get the index stock news save with stock dataframe
    # Get top 5 summary text
    # Merge into 1 text, preprocess
    # if merged text have lenght < max_input_lenght, add zero to it to meet the lenght
    # if larger, remove sentences until meet the lenght, than add zero
    # tokenize the text, convert tokens into ids
    # convert into (batch, 14, n) data
    max_string_lenght = 50
    model_name = "bert-base-uncased"
    sentence_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    file_path = './NLP/news_data/' + symbol + "/" + symbol + "_" + "data.csv"
    news_query_folder = "./NLP/news_query"
    news_query_file_name = symbol + "_" + "query.json"
    news_query_path = news_query_folder + "/" + news_query_file_name
    with open(news_query_path, "r") as f:
        queries = json.load(f)
    keyword_query = list(queries.values())
    # df = load_data_with_index(file_path, stock_df.index)
    news_df = pd.read_csv(file_path)
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df = filter_news_by_stock_dates(news_df, stock_df, max_rows=5)
    top_sentences_dict = []
    # news_df.reset_index(drop=True, inplace=True)
    for date in news_df["date"].unique():
        summmary_columns = news_df[news_df["date"] == date]["summary"]
        top_summary = get_similar_summary(summmary_columns.values, keyword_query, sentence_model, topK, 0.5)
        flattened_array = np.ravel(np.array(top_summary))
        merged_summary = ' '.join(flattened_array)
        ids = sentence_tokenize(merged_summary, sentence_model)
        top_sentences_dict.append(ids)

    top_sentences_dict = np.array(top_sentences_dict)
    data = prepare_time_series_news_data(top_sentences_dict, window_size, output_step, 1)

    return data


def prepare_raw_news_data(stock_df, symbol, window_size, from_date, to_date, output_step, topK, new_data=False):
    # Read the csv file
    # Get the index stock news save with stock dataframe
    # Get top 5 summary text
    # Merge into 1 text, preprocess
    # if merged text have lenght < max_input_lenght, add zero to it to meet the lenght
    # if larger, remove sentences until meet the lenght, than add zero
    # tokenize the text, convert tokens into ids
    # convert into (batch, 14, n) data
    max_string_lenght = 1000

    file_path = './NLP/news_data/' + symbol + "/" + symbol + "_" + "data.csv"
    news_query_folder = "./NLP/news_query"
    news_query_file_name = symbol + "_" + "query.json"
    news_query_path = news_query_folder + "/" + news_query_file_name
    with open(news_query_path, "r") as f:
        queries = json.load(f)
    keyword_query = list(queries.values())
    df = load_data_with_index(file_path, stock_df.index)
    top_sentences_dict = []
    for index in df.index:

        summary_columns = df.loc[index, "summary"]
        if isinstance(summary_columns, str):
            top_sentences = summary_columns[:max_string_lenght]
            top_sentences_dict.append(top_sentences)
    top_sentences_dict = np.array(top_sentences_dict)
    data = prepare_time_series_news_raw_data(top_sentences_dict, window_size, output_step, 1)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            data[i, j] = f"string_{i}_{j}"

    # Split the strings into a third dimension
    array_3d = np.array([s.split("_") for s in data.flat]).reshape((data.shape[0], data.shape[1], -1))
    return array_3d


# define function to load csv file with given index

import numpy as np


def load_data_with_index(csv_file, index):
    df = pd.read_csv(csv_file)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    index = pd.to_datetime(index)

    not_found = [idx for idx in index if idx not in df.index]

    if not_found:
        print(f"The following index values are not found in the DataFrame: {not_found}")
        new_rows = []
        for idx in not_found:
            nearest_rows = df.index[np.abs(df.index - idx).argsort()][:5]
            if len(nearest_rows) > 0:
                for row in nearest_rows:
                    new_rows.append(df.loc[row].to_dict())
        new_rows = pd.DataFrame(new_rows)
        new_rows.set_index(index[:len(new_rows)], inplace=True)
        df = pd.concat([df, new_rows])

    df = df.groupby(df.index).first()  # Keep only the first occurrence of each index label
    df = df.reindex(index)
    return df


def tokenize(text, tokenizer):
    text = preprocess_text(text)
    # Tokenize the text
    tokens = tokenizer.tokenize(text)

    # Remove stopwords
    stopwords = tokenizer.get_added_vocab()
    filtered_tokens = [token for token in tokens if token not in stopwords]

    # Perform lemmatization
    lemmatized_tokens = [tokenizer.decode(tokenizer.encode(token, add_special_tokens=False)) for token in
                         filtered_tokens]

    # Convert tokens to IDs
    token_ids = tokenizer.convert_tokens_to_ids(lemmatized_tokens)
    # Convert IDs back to text
    # processed_text = tokenizer.decode(token_ids)
    return token_ids, lemmatized_tokens


def sentence_tokenize(text, sentence_model):
    # Encode the sentences using the sentence model
    sentence_embeddings = sentence_model.encode(text)

    return sentence_embeddings


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()

    # Remove non-text elements
    text = re.sub(r'[^\w\s.]', '', text)

    text = re.sub(r'[ \t]+(?=\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text


def extract_sentences(text, queries):
    # Split the text into sentences

    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)

    # Create an empty list to store the matching sentences
    matching_sentences = []

    # Loop through each sentence
    for sentence in sentences:
        # Check if any of the query words are in the sentence
        if any(query.lower() in sentence for query in queries):
            # If a query word is found, add the sentence to the list of matching sentences
            matching_sentences.append(sentence)

    return matching_sentences


def get_similarity_score(sentence, queries, sentence_model):
    sentence_embeddings = sentence_model.encode(sentence)
    sim_list = []
    for query in queries:
        query_embeddings = sentence_model.encode(query).reshape(1, -1)
        similarity = cosine_similarity(sentence_embeddings, query_embeddings)
        sim_list.append(similarity)

    similarity = sum(sim_list) / len(sim_list)
    return similarity


def get_similar_sentences(paragraph, queries, sentence_model, threshold=0.0):
    if isinstance(paragraph, list):
        # Split the paragraph into individual sentences
        sentences = extract_sentences(paragraph, queries)

        # Calculate the similarity scores using vectorized operations
        # similarity_scores = np.array([get_similarity_score(sentence, queries, sentence_model) for sentence in sentences])
        similarity_scores = np.array(get_similarity_score(sentences, queries, sentence_model))

        sorted_indices = np.argsort(similarity_scores, axis=0)[::-1]
        sorted_sentences = sentences[sorted_indices]

        return sorted_sentences
    elif isinstance(paragraph, str):
        # Split the paragraph into individual sentences
        sentences = np.array(extract_sentences(paragraph, queries), dtype=str)
        if len(sentences) > 0:
            # Calculate the similarity scores using vectorized operations
            # similarity_scores = np.array([get_similarity_score(sentence, queries, sentence_model) for sentence in sentences])
            similarity_scores = np.array(get_similarity_score(sentences, queries, sentence_model))

            sorted_indices = np.argsort(similarity_scores, axis=0)[::-1]
            sorted_indices = sorted_indices.flatten().astype(int)
            sorted_sentences = sentences[sorted_indices]
            return sorted_sentences.tolist()
        else:
            return []


def get_similar_summary(summaries, queries, sentence_model, topK, threshold=0.5):
    # Calculate the similarity scores using vectorized operations
    if isinstance(summaries, list) or isinstance(summaries, np.ndarray):
        similarity_scores = np.array(get_similarity_score(summaries, queries, sentence_model))
        sorted_indices = np.argsort(similarity_scores, axis=0)[::-1]
        sorted_sentences = summaries[sorted_indices]
        return sorted_sentences[:topK]
    elif isinstance(summaries, str):
        return summaries


# If there isn't any file in news_web_url/ + symbol /+ symbol_url.csv than:
# Download news from all source with from_Date of "2022-07-01" and to_date is Now
# Merge them with unique URL
# Save in the news_web_url/ + symbol /+ symbol_url.csv
# Else:
# read the file in news_web_url/ + symbol /+ symbol_url.csv
# to_date is Now, from_date is to_date - window_size
# concat the dataframe into the main csv file and filter by url
# while also return the dataframe of the window size news


def download_nyt_news(query, folder, from_date, to_date, page_size):
    # Subtract the timedelta object from the from_date to get the to_date
    # api_key = 'fa24ffdbf32f4feeb3ef755fad66a2bd'
    api_key = '8rcgDc65Yn8HCXG68vhA42bvawaKJ8xk'
    date_format = '%Y-%m-%d'
    if to_date is None:
        to_date = datetime.now()
        to_date = to_date.strftime(date_format)
    # else:
    #     to_date = datetime.strptime(to_date, date_format)

    # from_date = datetime.strptime(from_date, date_format)

    # Define an empty list to store the results
    results = []

    # Define a counter to keep track of the number of articles retrieved
    count = 0

    # Define the API endpoint
    endpoint = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    for page in range(0, page_size):
        # Define the query parameters
        params = {
            'q': query,
            'begin_date': from_date,
            'end_date': to_date,
            'page': page,
            'api-key': "fs5aF1Yb7hl5u7L3AZDrAAZn125qhxv8"
        }

        # Send the HTTP GET request to the API endpoint and retrieve the response
        response = requests.get(endpoint, params=params)

        # Check if the response was successful
        if response.status_code != 200:
            print(f"Error retrieving news articles: {response.text}")
            break

        # Parse the JSON response
        data = json.loads(response.text)

        # Check if there are no more articles to retrieve
        if not data['response']['docs']:
            break
        data = data['response']['docs']
        # Loop through the articles and extract the relevant information
        for article in data:
            # Check if the article has a valid publication date
            if 'pub_date' not in article or not article['pub_date']:
                continue

            # Convert the publication date string to a datetime object
            pub_date = datetime.strptime(article['pub_date'], '%Y-%m-%dT%H:%M:%S%z').date()
            # Check if the article's _id already exists in results
            article_id = article.get('_id')
            if article_id and any(a.get('_id') == article_id for a in results):
                continue
            # Add the relevant information to the results list
            results.append(article)

    # Save the results to a CSV file
    folder_path = "./news_data/" + folder + "/"
    filename = "nyt_news.csv"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Check if file with the given name exists in the folder
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        # If it does, find an available numeric prefix
        i = 1
        while True:
            new_filename = f"{i}_{filename}"
            new_file_path = os.path.join(folder_path, new_filename)
            if not os.path.exists(new_file_path):
                break
            i += 1
        # Rename the file with the new prefixed filename
        os.rename(file_path, new_file_path)
        print(f"Renamed file {filename} to {new_filename}")
        # Update the file path to the new prefixed filename
        file_path = new_file_path
    # Save the articles to a CSV file
    df = pd.DataFrame.from_records(results)
    df = df.rename(columns={'pub_date': 'date'})
    df = df.rename(columns={'web_url': 'url'})
    # Convert the `datetime` column to a pandas datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Set the `date` column as the index
    df.set_index('date', inplace=True)
    df.to_csv(file_path, index=False)


def increase_n_days(date, n_days):
    date_obj = datetime.strptime(date, '%Y-%m-%d').date()
    delta = timedelta(days=n_days)
    date_obj += delta
    return date_obj.strftime('%Y-%m-%d')


def download_nyt_news(query, folder, from_date, to_date, page_size):
    # Subtract the timedelta object from the from_date to get the to_date
    # api_key = 'fa24ffdbf32f4feeb3ef755fad66a2bd'
    api_key = '8rcgDc65Yn8HCXG68vhA42bvawaKJ8xk'
    date_format = '%Y-%m-%d'
    if to_date is None:
        to_date = datetime.now()
        to_date = to_date.strftime(date_format)
    # else:
    #     to_date = datetime.strptime(to_date, date_format)

    # from_date = datetime.strptime(from_date, date_format)

    # Define an empty list to store the results
    results = []

    # Define a counter to keep track of the number of articles retrieved
    count = 0

    # Define the API endpoint
    endpoint = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    for page in range(0, page_size):
        # Define the query parameters
        params = {
            'q': query,
            'begin_date': from_date,
            'end_date': to_date,
            'page': page,
            'api-key': "fs5aF1Yb7hl5u7L3AZDrAAZn125qhxv8"
        }

        # Send the HTTP GET request to the API endpoint and retrieve the response
        response = requests.get(endpoint, params=params)

        # Check if the response was successful
        if response.status_code != 200:
            print(f"Error retrieving news articles: {response.text}")
            break

        # Parse the JSON response
        data = json.loads(response.text)

        # Check if there are no more articles to retrieve
        if not data['response']['docs']:
            break
        data = data['response']['docs']
        # Loop through the articles and extract the relevant information
        for article in data:
            # Check if the article has a valid publication date
            if 'pub_date' not in article or not article['pub_date']:
                continue

            # Convert the publication date string to a datetime object
            pub_date = datetime.strptime(article['pub_date'], '%Y-%m-%dT%H:%M:%S%z').date()
            # Check if the article's _id already exists in results
            article_id = article.get('_id')
            if article_id and any(a.get('_id') == article_id for a in results):
                continue
            # Add the relevant information to the results list
            results.append(article)

    # Save the results to a CSV file
    folder_path = "./news_data/" + folder + "/"
    filename = "nyt_news.csv"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    # Check if file with the given name exists in the folder
    file_path = os.path.join(folder_path, filename)
    if os.path.exists(file_path):
        # If it does, find an available numeric prefix
        i = 1
        while True:
            new_filename = f"{i}_{filename}"
            new_file_path = os.path.join(folder_path, new_filename)
            if not os.path.exists(new_file_path):
                break
            i += 1
        # Rename the file with the new prefixed filename
        os.rename(file_path, new_file_path)
        print(f"Renamed file {filename} to {new_filename}")
        # Update the file path to the new prefixed filename
        file_path = new_file_path
    # Save the articles to a CSV file
    df = pd.DataFrame.from_records(results)
    df = df.rename(columns={'pub_date': 'date'})
    df = df.rename(columns={'web_url': 'url'})
    # Convert the `datetime` column to a pandas datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Set the `date` column as the index
    df.set_index('date', inplace=True)
    df.to_csv(file_path, index=False)


def download_finhub_news(symbol, from_date, to_date, save_folder, new_data):
    filename = f'{symbol}_finhub_url.csv'
    filepath = os.path.join(save_folder, filename)
    if os.path.exists(filepath):
        if not new_data:
            print("File already exists. Skipping download.")
            df = pd.read_csv(filepath, index_col='date')
            return df
        else:
            stop_date = datetime.strptime(to_date, "%Y-%m-%d")
            from_date_dt = datetime.strptime(from_date, "%Y-%m-%d")

            final_df = pd.DataFrame()
            finnhub_client = finnhub.Client(api_key="chdrr1pr01qi6ghjsatgchdrr1pr01qi6ghjsau0")
            while from_date_dt <= stop_date:
                to_date = increase_n_days(from_date, 10)
                df = pd.DataFrame(finnhub_client.company_news(symbol, _from=from_date, to=to_date))
                df['datetime'] = df['datetime'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
                df = df.sort_values('datetime')
                df.drop_duplicates(subset='id', keep='first', inplace=True)
                final_df = pd.concat([final_df, df])
                from_date_dt = datetime.strptime(to_date, "%Y-%m-%d")
                from_date = from_date_dt.strftime('%Y-%m-%d')
            final_df.drop_duplicates(subset='id', keep='first', inplace=True)
            final_df = final_df.rename(columns={'datetime': 'date'})
            # Convert the `datetime` column to a pandas datetime format
            final_df['date'] = pd.to_datetime(final_df['date'])
            # # Set the `datetime` column as the index
            final_df.set_index('date', inplace=True)
            os.makedirs(save_folder, exist_ok=True)
            # Export the DataFrame to a CSV file
            final_df.to_csv(filepath, index=True)
            return final_df
    else:
        stop_date = datetime.strptime(to_date, "%Y-%m-%d")
        from_date_dt = datetime.strptime(from_date, "%Y-%m-%d")

        final_df = pd.DataFrame()
        finnhub_client = finnhub.Client(api_key="chdrr1pr01qi6ghjsatgchdrr1pr01qi6ghjsau0")
        while from_date_dt <= stop_date:
            to_date = increase_n_days(from_date, 10)
            df = pd.DataFrame(finnhub_client.company_news(symbol, _from=from_date, to=to_date))
            df['datetime'] = df['datetime'].apply(lambda x: datetime.fromtimestamp(x).strftime('%Y-%m-%d'))
            df = df.sort_values('datetime')
            df.drop_duplicates(subset='id', keep='first', inplace=True)
            final_df = pd.concat([final_df, df])
            from_date_dt = datetime.strptime(to_date, "%Y-%m-%d")
            from_date = from_date_dt.strftime('%Y-%m-%d')
        final_df.drop_duplicates(subset='id', keep='first', inplace=True)
        final_df = final_df.rename(columns={'datetime': 'date'})
        # Convert the `datetime` column to a pandas datetime format
        final_df['date'] = pd.to_datetime(final_df['date'])
        # # Set the `datetime` column as the index
        final_df.set_index('date', inplace=True)
        os.makedirs(save_folder, exist_ok=True)
        # Export the DataFrame to a CSV file
        final_df.to_csv(filepath, index=True)
        return final_df


def download_benzinga_news(symbol, from_date, to_date, save_folder, new_data):
    api_key = '936c241cdb1b4620b3bad6c77ba3ae4b'
    endpoint = 'https://api.benzinga.com/api/v2/news'

    # Set up query parameters
    params = {
        'accept': "application/json (default)",
        'tickers': symbol,  # Symbol passed as argument
        'token': api_key,
        'dateFrom': from_date,
        'dateTo': to_date,
        'pageSize': 100  # Example: Number of results per page
    }

    # Make the API request
    response = requests.get(endpoint, params=params)
    data = xmltodict.parse(response.text)

    # Convert the XML response to JSON
    json_data = json.dumps(data)
    articles = json.loads(json_data)['result']['item']

    # Process the response
    if response.status_code == 200:
        # Extract and save relevant information
        filename = f'{symbol}_benzinga_news.csv'
        filepath = os.path.join(save_folder, filename)

        df = pd.DataFrame(articles)
        df = df.rename(columns={'created': 'date'})
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df.set_index('date', inplace=True)

        os.makedirs(save_folder, exist_ok=True)

        if os.path.exists(filepath):
            if not new_data:
                print("File already exists. Skipping download.")
                df_existing = pd.read_csv(filepath, index_col='date')
                return pd.concat([df_existing, df])

        df.to_csv(filepath, index=True)
        return df
    else:
        print("Error occurred while fetching news:", data['error'])


def download_google_news(symbol, from_date, to_date, save_folder, new_data):
    filename = f'{symbol}_google_url.csv'
    filepath = os.path.join(save_folder, filename)
    if os.path.exists(filepath):
        if not new_data:
            print("File already exists. Skipping download.")
            df = pd.read_csv(filepath, index_col='date')
            return df
        else:
            # Subtract the timedelta object from the from_date to get the to_date
            # Create a new GoogleNews object
            googlenews = GoogleNews(lang='en')
            # Set the search query and time range
            googlenews.search(symbol)
            googlenews.set_time_range(from_date, to_date)
            # Retrieve the news articles
            articles = googlenews.result()

            df = pd.DataFrame.from_records(articles)
            df = df.drop(columns=['date'])
            df = df.rename(columns={'link': 'url'})
            df = df.rename(columns={'datetime': 'date'})
            df = df.rename(columns={'media': 'source'})
            # Convert the `datetime` column to a pandas datetime format
            df['date'] = pd.to_datetime(df['date'])
            # Convert 'date' to the format 'year-month-day'
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            # Set the `datetime` column as the index
            df.set_index('date', inplace=True)
            # Create the folder if it doesn't exist
            os.makedirs(save_folder, exist_ok=True)
            # Save the DataFrame to the specified path
            df.to_csv(filepath, index=True)
            return df
    else:
        # Subtract the timedelta object from the from_date to get the to_date
        # Create a new GoogleNews object
        googlenews = GoogleNews(lang='en')
        # Set the search query and time range
        googlenews.search(symbol)
        googlenews.set_time_range(from_date, to_date)
        # Retrieve the news articles
        articles = googlenews.result()

        df = pd.DataFrame.from_records(articles)
        df = df.drop(columns=['date'])
        df = df.rename(columns={'link': 'url'})
        df = df.rename(columns={'datetime': 'date'})
        df = df.rename(columns={'media': 'source'})
        # Convert the `datetime` column to a pandas datetime format
        df['date'] = pd.to_datetime(df['date'])
        # Convert 'date' to the format 'year-month-day'
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        # Set the `datetime` column as the index
        df.set_index('date', inplace=True)
        # Create the folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)
        # Save the DataFrame to the specified path
        df.to_csv(filepath, index=True)
        return df


def download_alpha_vantage_news(symbol, from_date, to_date, save_folder, new_data):
    filename = f'{symbol}_alpha_vantage_url.csv'
    filepath = os.path.join(save_folder, filename)
    if os.path.exists(filepath):
        if not new_data:
            print("File already exists. Skipping download.")
            df = pd.read_csv(filepath, index_col='date')
            return df
        else:
            # Subtract the timedelta object from the from_date to get the to_date
            # Create a new AlphaVantage object or use the appropriate library

            # Set up the necessary parameters for Alpha Vantage API
            api_key = 'XOLA7URKCZHU7C9X'  # Replace with your actual API key
            api_endpoint = 'https://www.alphavantage.co/query'

            # Define the API parameters
            params = {
                'function': 'NEWS_SENTIMENT',
                'tickers': symbol,
                'sort': "EARLIEST",
                'limit': '5',
                'apikey': api_key
            }

            # Convert from_date to datetime object
            from_date_obj = datetime.strptime(from_date, '%Y-%m-%d')

            # Get the current date
            current_date_obj = datetime.strptime(to_date, '%Y-%m-%d')

            articles = []  # List to store all news articles

            # Loop through each day
            while from_date_obj <= current_date_obj:
                # Format the current date as time_from and time_to
                params['time_from'] = from_date_obj.strftime('%Y%m%dT%H%M')
                params['time_to'] = from_date_obj.strftime('%Y%m%dT%H%M')

                # Make the API request to get the daily news articles
                response = requests.get(api_endpoint, params=params)
                data = response.json()
                articles.extend(data['articles'])

                from_date_obj += timedelta(days=1)

            # Convert the articles list into a DataFrame
            df = pd.DataFrame.from_records(articles)
            df = df.drop(columns=['date'])
            df = df.rename(columns={'link': 'url'})
            df = df.rename(columns={'datetime': 'date'})
            df = df.rename(columns={'media': 'source'})
            # Convert the `date` column to a pandas datetime format
            df['date'] = pd.to_datetime(df['date'])
            # Convert 'date' to the format 'year-month-day'
            df['date'] = df['date'].dt.strftime('%Y-%m-%d')
            # Set the `date` column as the index
            df.set_index('date', inplace=True)
            os.makedirs(save_folder, exist_ok=True)
            # Save the DataFrame to the specified path
            df.to_csv(filepath, index=True)
            return df
    else:
        # Subtract the timedelta object from the from_date to get the to_date
        # Create a new AlphaVantage object or use the appropriate library

        # Set up the necessary parameters for Alpha Vantage API
        api_key = 'XOLA7URKCZHU7C9X'  # Replace with your actual API key
        api_endpoint = 'https://www.alphavantage.co/query'

        # Define the API parameters
        params = {
            'function': 'NEWS_SENTIMENT',
            'tickers': symbol,
            'sort': "EARLIEST",
            'limit': '5',
            'apikey': api_key
        }

        # Convert from_date to datetime object
        from_date_obj = datetime.strptime(from_date, '%Y-%m-%d')

        # Get the current date
        current_date_obj = datetime.strptime(to_date, '%Y-%m-%d')

        articles = []  # List to store all news articles

        # Loop through each day
        while from_date_obj <= current_date_obj:
            # Format the current date as time_from and time_to
            params['time_from'] = from_date_obj.strftime('%Y%m%dT%H%M')
            params['time_to'] = (from_date_obj + timedelta(days=3)).strftime('%Y%m%dT%H%M')

            # Make the API request to get the daily news articles
            response = requests.get(api_endpoint, params=params)
            data = response.json()
            articles.extend(data['feed'])

            from_date_obj += timedelta(days=3)

        # Convert the articles list into a DataFrame
        df = pd.DataFrame.from_records(articles)
        df = df.rename(columns={'time_published': 'date'})
        # Convert the `date` column to a pandas datetime format
        df['date'] = pd.to_datetime(df['date'])
        # Convert 'date' to the format 'year-month-day'
        df['date'] = df['date'].dt.strftime('%Y-%m-%d')
        # Set the `date` column as the index
        df.set_index('date', inplace=True)
        os.makedirs(save_folder, exist_ok=True)
        # Save the DataFrame to the specified path
        df.to_csv(filepath, index=True)
        return df


def download_bing_news(symbol, from_date, to_date, save_folder, new_data):
    filename = f'{symbol}_bing_url.csv'
    filepath = os.path.join(save_folder, filename)
    if os.path.exists(filepath):
        if not new_data:
            print("File already exists. Skipping download.")
            df = pd.read_csv(filepath, index_col='date')
            return df
        else:
            # Perform the Bing News API request
            news_list = get_bing_news(symbol, from_date, to_date)

            # Convert the news list to a DataFrame
            df = pd.DataFrame(news_list)
            df.set_index('date', inplace=True)

            # Save the DataFrame to the specified path
            df.to_csv(filepath, index=True)
            return df
    else:
        # Perform the Bing News API request
        news_list = get_bing_news(symbol, from_date, to_date)

        # Convert the news list to a DataFrame
        df = pd.DataFrame(news_list)
        df.set_index('date', inplace=True)

        # Create the folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=True)

        # Save the DataFrame to the specified path
        df.to_csv(filepath, index=True)
        return df


def get_bing_news(symbol, from_date, to_date):
    # Define the Bing News API endpoint
    api_endpoint = 'https://api.bing.microsoft.com/v7.0/news/search'

    # Set up the request headers with the subscription key
    headers = {
        "X-BingApis-SDK": "true",
        "X-RapidAPI-Key": "09bd01b3eemsh2b008d6bf606fc4p166c60jsnf4070b8bf6eb",
        "X-RapidAPI-Host": "bing-news-search1.p.rapidapi.com"
    }

    # Set the query parameters
    params = {
        'q': symbol,
        'count': 10,
        'freshness': 'Day',
        'mkt': 'en-US',
        'safeSearch': 'Off',
        'fromDate': from_date,
        'toDate': to_date
    }

    # Send the GET request to the API
    response = requests.get(api_endpoint, headers=headers, params=params)
    data = response.json()

    # Extract the articles from the response
    articles = data.get('value', [])

    # Process the articles as needed
    news_list = []
    for article in articles:
        title = article.get('name')
        description = article.get('description')
        url = article.get('url')
        published_at = article.get('datePublished')

        # Add the article details to the list
        news_list.append({
            'title': title,
            'description': description,
            'url': url,
            'published_at': published_at
        })

    return news_list


def download_news(symbol, from_date, window_size, new_data=True):
    to_date = datetime.now().strftime('%Y-%m-%d')
    save_folder = f'./NPL/news_web_url/{symbol}/'
    file_name = f'{symbol}_url.csv'
    save_path = os.path.join(save_folder, file_name)
    main_df = pd.DataFrame()
    if not os.path.exists(save_folder + file_name):
        # Download news from all sources with the specified date range
        google_df = download_google_news(symbol, from_date, to_date, save_folder, new_data)
        main_df = pd.concat([main_df, google_df])
        finhub_df = download_finhub_news(symbol, from_date, to_date, save_folder, new_data=False)
        main_df = pd.concat([main_df, finhub_df])
        benzema_df = download_benzinga_news(symbol, from_date, to_date, save_folder, new_data=False)
        main_df = pd.concat([main_df, benzema_df])

        # alphavantage_df = download_alpha_vantage_news(symbol, from_date, to_date, save_folder, new_data)
        # bing_df = download_bing_news(symbol, from_date, to_date, save_folder, new_data = True)
        # main_df = pd.concat([main_df, bing_df])

        main_df.to_csv(save_path, index=True)
        # Filter the dataframe by URL
        window_df = main_df[main_df['url'].isin(main_df['url'].unique())]

        # Save the updated dataframe to the CSV file
        main_df.to_csv(save_path, index=True)
    else:
        # # Read the existing CSV file
        # main_df = pd.read_csv(save_path)

        # # Set the new date range
        # from_date = (datetime.strptime(to_date, '%Y-%m-%d') - timedelta(days=window_size)).strftime('%Y-%m-%d')

        # # Download news from additional sources
        # additional_df = download_additional_sources_news(from_date, to_date)

        # # Concatenate the dataframes
        # main_df = pd.concat([main_df, additional_df])

        # # Filter the dataframe by URL
        # window_df = main_df[main_df['URL'].isin(main_df['URL'].unique())]

        # # Save the updated dataframe to the CSV file
        # main_df.to_csv(save_path, index=False)
        pass
    return main_df.values

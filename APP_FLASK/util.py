import numpy as np
from sentence_transformers import SentenceTransformer, util
import json
import pandas as pd
import util as u
from NLP import util as nlp_u
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
def prepare_stock_data(data, window_size):
    features = data.shape[-1]
    n_samples = (len(data) - (window_size - 1))
    X = np.zeros((n_samples, window_size, features))
    for i in range(n_samples):
        for j in range(window_size):
            X[i][j] = (data[i + (j)])
    timeseries_price = X[:, :5]
    timeseries_stock = X[:, 5:]
    return timeseries_price, timeseries_stock

def prepare_news_data(stock_df, symbol, window_size, from_date, to_date, output_step, topK, max_string_length, new_data=False):
    file_name = f'./realtime_data/news_{symbol}_w{window_size}_l{max_string_length}.npy'
    if new_data == True:
        model_name = "ProsusAI/finbert"
        tokenizer = AutoTokenizer.from_pretrained(model_name) #### 
        sentence_model = SentenceTransformer(model_name)
        file_path = './NLP/news_data/' + symbol + "/" + symbol + "_" + "data.csv"
        news_query_folder = "./NLP/news_query"
        news_query_file_name = symbol + "_" + "query.json"
        news_query_path = news_query_folder + "/" + news_query_file_name

        with open(news_query_path, "r") as f:
            queries = json.load(f)
        keyword_query = list(queries.values())

        news_df = pd.read_csv(file_path)
        news_df['date'] = pd.to_datetime(news_df['date'])
        news_df = filter_news_by_stock_dates(news_df, stock_df, max_rows=5)
        top_sentences_dict = []
        total_lenght = 0
        
        for date in news_df["date"].unique():
            summary_columns = news_df[news_df["date"] == date]["summary"]
            top_summary = get_similar_summary(summary_columns.values, keyword_query, sentence_model, topK, 0.5)
            flattened_array = np.ravel(np.array(top_summary))
            merged_summary = ' '.join(flattened_array)[:max_string_length]
            total_lenght += len(merged_summary)
            # print("len", len(merged_summary))
            encoded_summary = tokenizer.encode_plus(
                merged_summary,
                max_length=max_string_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt"
            )
            input_ids = encoded_summary["input_ids"]
            embeddings = sentence_model.encode(merged_summary)

            top_sentences_dict.append(embeddings)
        print("average lenght: ", total_lenght / len(news_df["date"].unique()))
        
        top_sentences_dict = np.array(top_sentences_dict)
        if os.path.exists(file_name):
            os.remove(file_name)
        np.save(file_name, top_sentences_dict)
    else:
        if os.path.exists(file_name):
            top_sentences_dict = np.load(file_name, allow_pickle=True)
        else:
            model_name = "ProsusAI/finbert"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            sentence_model = SentenceTransformer(model_name)
            file_path = './NLP/news_data/' + symbol + "/" + symbol + "_" + "data.csv"
            news_query_folder = "./NLP/news_query"
            news_query_file_name = symbol + "_" + "query.json"
            news_query_path = news_query_folder + "/" + news_query_file_name

            with open(news_query_path, "r") as f:
                queries = json.load(f)
            keyword_query = list(queries.values())

            news_df = pd.read_csv(file_path)
            news_df['date'] = pd.to_datetime(news_df['date'])
            news_df = filter_news_by_stock_dates(news_df, stock_df, max_rows=5)
            top_sentences_dict = []
            total_lenght = 0
            
            for date in news_df["date"].unique():
                summary_columns = news_df[news_df["date"] == date]["summary"]
                top_summary = get_similar_summary(summary_columns.values, keyword_query, sentence_model, topK, 0.5)
                flattened_array = np.ravel(np.array(top_summary))
                merged_summary = ' '.join(flattened_array)[:max_string_length]
                total_lenght += len(merged_summary)
                # print("len", len(merged_summary))
                encoded_summary = tokenizer.encode_plus(
                    merged_summary,
                    max_length=max_string_length,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                input_ids = encoded_summary["input_ids"]
                embeddings = sentence_model.encode(merged_summary)

                top_sentences_dict.append(embeddings)
                print("average lenght: ", total_lenght / len(news_df["date"].unique()))
            
            top_sentences_dict = np.array(top_sentences_dict)           
            
            if os.path.exists(file_name):
                os.remove(file_name)
            np.save(file_name, top_sentences_dict)
    data = prepare_time_series_news_data(top_sentences_dict, window_size, output_step, 1)

    return data, top_sentences_dict
# def prepare_news_data(stock_df, symbol, window_size, topK, new_data=False):
#     # Read the csv file
#     # Get the index stock news save with stock dataframe
#     # Get top 5 summary text
#     # Merge into 1 text, preprocess
#     # if merged text have lenght < max_input_lenght, add zero to it to meet the lenght
#     # if larger, remove sentences until meet the lenght, than add zero
#     # tokenize the text, convert tokens into ids
#     # convert into (batch, 14, n) data
#     max_string_lenght = 50
#     model_name = "bert-base-uncased"
#     sentence_model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
#     file_path = './NLP/news_data/' + symbol + "/" + symbol + "_" + "data.csv"
#     news_query_folder = "./NLP/news_query"
#     news_query_file_name = symbol + "_" + "query.json"
#     news_query_path = news_query_folder + "/" + news_query_file_name
#     with open(news_query_path, "r") as f:
#         queries = json.load(f)
#     keyword_query = list(queries.values())
#     # df = load_data_with_index(file_path, stock_df.index)
#     news_df = pd.read_csv(file_path)
#     news_df['date'] = pd.to_datetime(news_df['date'])
#     news_df = nlp_u.filter_news_by_stock_dates(news_df, stock_df, max_rows=5)
#     top_sentences_dict = []
#     # news_df.reset_index(drop=True, inplace=True)
#     for date in news_df["date"].unique():
#         summmary_columns = news_df[news_df["date"] == date]["summary"]
#         top_summary = nlp_u.get_similar_summary(summmary_columns.values, keyword_query, sentence_model, topK, 0.5)
#         flattened_array = np.ravel(np.array(top_summary))
#         merged_summary = ' '.join(flattened_array)
#         ids = nlp_u.sentence_tokenize(merged_summary, sentence_model)
#         top_sentences_dict.append(ids)

#     top_sentences_dict = np.array(top_sentences_dict)
#     data = top_sentences_dict[-window_size:, :]
#     return data
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
                while news_counts[date] < rows_remaining and previous_date - pd.DateOffset(days=days_skipped) in \
                        news_df["date"]:
                    previous_date -= pd.DateOffset(days=days_skipped)
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
            while news_counts[
                date] < rows_remaining:  # and previous_date - pd.DateOffset(days=days_skipped) in news_df["date"]:
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


def get_similar_summary(summaries, queries, sentence_model, topK, threshold=0.5):
    # Calculate the similarity scores using vectorized operations
    if isinstance(summaries, list) or isinstance(summaries, np.ndarray):
        similarity_scores = np.array(get_similarity_score(summaries, queries, sentence_model))
        sorted_indices = np.argsort(similarity_scores, axis=0)[::-1]
        sorted_sentences = summaries[sorted_indices]
        return sorted_sentences
    elif isinstance(summaries, str):
        return summaries
def get_similarity_score(sentence, queries, sentence_model):
    sentence_embeddings = sentence_model.encode(sentence)
    sim_list = []
    for query in queries:
        query_embeddings = sentence_model.encode(query).reshape(1, -1)
        similarity = cosine_similarity(sentence_embeddings, query_embeddings)
        sim_list.append(similarity)

    similarity = sum(sim_list) / len(sim_list)
    return similarity
def prepare_time_series_news_data(data, window_size, output_step, dilation, stride=1):
    features = data.shape[-1]
    n_samples = (len(data) - dilation * (window_size - 1) - output_step)
    X = np.zeros((n_samples, window_size, features))
    features = len(data)

    for i in range(n_samples):
        for j in range(window_size):
            X[i][j] = (data[i + (j * dilation)])
    return X
import csv
import random
import logging
import collections

import numpy as np
import tensorflow as tf


logger = logging.getLogger("data_handler")
formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(filename)s:%(lineno)d:%(message)s')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

skip_key = ['Currency', 'Fiscal Period', 'Fiscal Year', 'Publish Date', 'Report Date', 'SimFinId',
            'Ticker', 'title_key']

category_key = ["IndustryId"]


def load_industry_cate(path):
    logger.info("Load indusrty category from: {}".format(path))
    mapping = {}
    with open(path, "r") as f:
        tokens = f.read().split("\n")

    for i, token in enumerate(tokens):
        mapping[token] = i

    return mapping


def load_value_keys(path):
    logger.info("Load value keys from: {}".format(path))
    with open(path, "r") as f:
        value_keys = f.read().split("\n")

    return value_keys


def load_csv_data(path, used_keys):
    logger.info("Load csv data from: {}".format(path))
    cate_key = "IndustryId"
    data = {}
    max_value = 0
    max_price = 0
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            row_dict = {}
            value_vector = []

            mean_price = float(row["mean_price"])
            # Skip negtive and outrange price
            if mean_price > 99999 or mean_price < 0:
                continue

            title_key = row["title_key"]
            mean_price = float(row["mean_price"])
            industry_id = row[cate_key]
            max_price = max(max_price, mean_price)
            for key in used_keys:
                value = row[key]
                if value:
                    value = float(value)
                else:
                    value = 0.

                if key == "mean_price":
                    continue
                else:
                    value_vector.append(value)
                    max_value = max(max_value, abs(value))

            row_dict["value_vector"] = value_vector
            row_dict["mean_price"] = [mean_price]
            row_dict["industry_id"] = industry_id

            data[title_key] = row_dict

    return data, max_value, max_price


def load_csv_trace_data(path, used_keys):
    logger.info("Load csv data from: {}".format(path))
    cate_key = "IndustryId"
    data = collections.defaultdict(list)
    max_value = 0
    max_price = 0
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile)
        for i, row in enumerate(reader):
            row_dict = {}
            value_vector = []

            mean_price = float(row["mean_price"])
            # Skip negtive and outrange price
            if mean_price > 9999 or mean_price < 0:
                continue

            ticker_key = row["Ticker"]
            title_key = row["title_key"]
            mean_price = float(row["mean_price"])
            industry_id = row[cate_key]
            max_price = max(max_price, mean_price)
            for key in used_keys:
                if key == "mean_price":
                    continue
                value = row[key]
                if value:
                    value = float(value)
                else:
                    value = 0.

                value_vector.append(value)
                max_value = max(max_value, abs(value))

            row_dict["value_vector"] = value_vector
            row_dict["mean_price"] = mean_price
            row_dict["industry_id"] = industry_id
            row_dict["title_key"] = title_key

            data[ticker_key].append(row_dict)

    sort_data = {}
    for ticker, rows in data.items():
        sort_row = sorted(rows, key=lambda x: x["title_key"])
        sort_data[ticker] = sort_row

    return sort_data, max_value, max_price


def add_padding(diff, value_list, price_list, industry_list, mask_list):
    value_pad = np.zeros_like(value_list[0])
    price_pad = 0
    industry_pad = 0
    mask_pad = np.ones_like(mask_list[0])

    value_list = [value_pad]*diff + value_list
    price_list = [price_pad]*diff + price_list
    industry_list = [industry_pad]*diff + industry_list
    mask_list = [mask_pad]*diff + mask_list

    return value_list, price_list, industry_list, mask_list


def get_predict_generator_fn(data, max_value, max_price, industry_mapping, seq_len):
    total_ticker = list(data.keys())

    log_max_value = np.log(max_value)

    logger.info("log_max_value: {:.2f} | max_price: {}".format(
            log_max_value, max_price))

    def gen_fn():
        for ticker in total_ticker:
            row_list = data[ticker]
            if len(row_list) < 4:
                continue

            current_price_list = []
            value_list = []
            price_list = []
            industry_list = []
            mask_list = []

            for row in row_list:

                value_vector = row["value_vector"]
                mean_price = row["mean_price"]
                industry_id = row["industry_id"]
                industry_int = industry_mapping[industry_id]

                if mean_price == 0.:
                    continue

                value_arr = np.array(value_vector)
                price_arr = np.array(mean_price)
                current_price_list.append(mean_price)

                # Rescale value_arr
                neg_sign = value_arr < 0
                neg_sign = neg_sign.astype(np.float32)
                value_arr = np.abs(value_arr)

                # Set ignore value less than 1.
                value_arr = value_arr - value_arr * (value_arr < 1).astype(np.float32)
                mask_arr = np.equal(value_arr, 0.).astype(np.float32)

                # Set 0 to 1 for log fn.
                value_arr = value_arr + mask_arr
                log_value_arr = np.log(value_arr)

                # Rescale value
                log_value_arr = log_value_arr / log_max_value
                price_arr = price_arr / max_price

                # Add sign infomation
                log_value_arr = np.stack([log_value_arr, neg_sign], axis=1)

                # Add mask for industry id
                mask_arr = np.concatenate([mask_arr, np.array([0]).astype(np.float32)])

                # Add arr to list
                value_list.append(log_value_arr)
                price_list.append(price_arr)
                industry_list.append(industry_int)
                mask_list.append(mask_arr)

            # Check list len, add padding or slice list.
            if len(value_list) == 0:
                continue
            if len(value_list) >= seq_len:
                value_list = value_list[-seq_len:]
                industry_list = industry_list[-seq_len:]
                mask_list = mask_list[-seq_len:]
                price_list = price_list[-seq_len:]
                trace_mask_list = [0] * seq_len
            elif len(value_list) < seq_len:
                # Add padding
                diff = seq_len - len(value_list)
                trace_mask_list = [1] * diff + [0] * len(value_list)
                value_list, price_list, industry_list, mask_list = add_padding(
                    diff, value_list, price_list, industry_list, mask_list)

            assert len(value_list) == seq_len
            assert len(price_list) == seq_len
            assert len(industry_list) == seq_len
            assert len(mask_list) == seq_len

            value_list_arr = np.array(value_list).astype(np.float32)
            industry_list_arr = np.array(industry_list).astype(np.int32)
            mask_list_arr = np.array(mask_list).astype(np.float32)
            price_list_arr = np.array(price_list).astype(np.float32)
            trace_mask_arr = np.array(trace_mask_list).astype(np.float32)

            # Calculate target
            current_mean = sum(current_price_list) / len(current_price_list)

            yield (value_list_arr, industry_list_arr, mask_list_arr, price_list_arr,
                   trace_mask_arr, current_mean, ticker)

    return gen_fn


def get_trace_generator_fn(data, max_value, max_price, industry_mapping, seq_len):
    total_ticker = list(data.keys())

    log_max_value = np.log(max_value)

    logger.info("log_max_value: {:.2f} | max_price: {}".format(
            log_max_value, max_price))

    def gen_fn():
        random.shuffle(total_ticker)

        for ticker in total_ticker:
            row_list = data[ticker]
            if len(row_list) < 5:
                continue

            future_data = row_list[-4:]
            future_prices = []
            for row in future_data:
                mean_price = row["mean_price"]
                if mean_price == 0.:
                    continue
                future_prices.append(mean_price)
            if len(future_prices) == 0:
                continue

            current_price_list = []
            value_list = []
            price_list = []
            industry_list = []
            mask_list = []

            for row in row_list[:-4]:

                value_vector = row["value_vector"]
                mean_price = row["mean_price"]
                industry_id = row["industry_id"]
                industry_int = industry_mapping[industry_id]

                if mean_price == 0.:
                    continue

                value_arr = np.array(value_vector)
                price_arr = np.array(mean_price)
                current_price_list.append(mean_price)

                # Rescale value_arr
                neg_sign = value_arr < 0
                neg_sign = neg_sign.astype(np.float32)
                value_arr = np.abs(value_arr)

                # Set ignore value less than 1.
                value_arr = value_arr - value_arr * (value_arr < 1).astype(np.float32)
                mask_arr = np.equal(value_arr, 0.).astype(np.float32)

                # Set 0 to 1 for log fn.
                value_arr = value_arr + mask_arr
                log_value_arr = np.log(value_arr)

                # Rescale value
                log_value_arr = log_value_arr / log_max_value
                price_arr = price_arr / max_price

                # Add sign infomation
                log_value_arr = np.stack([log_value_arr, neg_sign], axis=1)

                # Add mask for industry id
                mask_arr = np.concatenate([mask_arr, np.array([0]).astype(np.float32)])

                # Add arr to list
                value_list.append(log_value_arr)
                price_list.append(price_arr)
                industry_list.append(industry_int)
                mask_list.append(mask_arr)

            # Check list len, add padding or slice list.
            if len(value_list) == 0:
                continue
            if len(value_list) >= seq_len:
                value_list = value_list[-seq_len:]
                industry_list = industry_list[-seq_len:]
                mask_list = mask_list[-seq_len:]
                price_list = price_list[-seq_len:]
                trace_mask_list = [0] * seq_len
            elif len(value_list) < seq_len:
                # Add padding
                diff = seq_len - len(value_list)
                trace_mask_list = [1] * diff + [0] * len(value_list)
                value_list, price_list, industry_list, mask_list = add_padding(
                    diff, value_list, price_list, industry_list, mask_list)

            assert len(value_list) == seq_len
            assert len(price_list) == seq_len
            assert len(industry_list) == seq_len
            assert len(mask_list) == seq_len

            value_list_arr = np.array(value_list).astype(np.float32)
            industry_list_arr = np.array(industry_list).astype(np.int32)
            mask_list_arr = np.array(mask_list).astype(np.float32)
            price_list_arr = np.array(price_list).astype(np.float32)
            trace_mask_arr = np.array(trace_mask_list).astype(np.float32)

            # Calculate target
            future_mean = sum(future_prices) / len(future_prices)
            current_mean = sum(current_price_list) / len(current_price_list)
            diff = future_mean - current_mean
            # shift and scale to 0 ~ 1
            target = (diff / current_mean) / 2 + 0.5
            if target > 1:
                target = 1.

            yield (value_list_arr, industry_list_arr, mask_list_arr, price_list_arr,
                   trace_mask_arr, target)

    return gen_fn


def get_trace_dataset(generator_fn, data_type=None, data_shape=None):
    """Create a function to get dataset."""
    # value, industry_id, mask, mean price
    if not data_type:
        data_type = (tf.float32, tf.int32, tf.float32, tf.float32, tf.float32, tf.float32)
    if not data_shape:
        data_shape = (
            tf.TensorShape([None, None, None]), tf.TensorShape([None]),
            tf.TensorShape([None, None]), tf.TensorShape([None]), tf.TensorShape([None]),
            tf.TensorShape([])
        )
    dataset = tf.data.Dataset.from_generator(
        generator_fn, data_type, data_shape)

    return dataset


def get_generator_fn(data, max_value, max_price, industry_mapping):
    total_title = list(data.keys())

    log_max_value = np.log(max_value)

    logger.info("log_max_value: {:.2f} | max_price: {}".format(
            log_max_value, max_price))

    def gen_fn():
        random.shuffle(total_title)

        for title in total_title:
            row_data = data[title]
            value_vector = row_data["value_vector"]
            mean_price = row_data["mean_price"]
            industry_id = row_data["industry_id"]
            industry_int = industry_mapping[industry_id]

            if mean_price == 0.:
                continue

            value_arr = np.array(value_vector)
            price_arr = np.array(mean_price)

            # Rescale value_arr
            neg_sign = value_arr < 0
            neg_sign = neg_sign.astype(np.float32)
            value_arr = np.abs(value_arr)

            # Set ignore value less than 1.
            value_arr = value_arr - value_arr * (value_arr < 1).astype(np.float32)
            mask_arr = np.equal(value_arr, 0.).astype(np.float32)

            # Set 0 to 1 for log fn.
            value_arr = value_arr + mask_arr
            log_value_arr = np.log(value_arr)

            # Rescale value
            log_value_arr = log_value_arr / log_max_value
            price_arr = price_arr / max_price

            # Add sign infomation
            log_value_arr = np.stack([log_value_arr, neg_sign], axis=1)

            # Add mask for industry id
            mask_arr = np.concatenate([mask_arr, np.array([0]).astype(np.float32)])

            yield log_value_arr, industry_int, mask_arr, price_arr

    return gen_fn


def get_dataset(generator_fn, data_type=None, data_shape=None):
    """Create a function to get dataset."""
    # value_vector, mean_price
    if not data_type:
        data_type = (tf.float32, tf.int32, tf.float32, tf.float32)
    if not data_shape:
        data_shape = (
            tf.TensorShape([None, None]), tf.TensorShape([]),
            tf.TensorShape([None]), tf.TensorShape([None])
        )
    dataset = tf.data.Dataset.from_generator(
        generator_fn, data_type, data_shape)

    return dataset

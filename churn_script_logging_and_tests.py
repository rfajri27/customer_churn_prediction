"""
Author: Rahmat Fajri
Date: 6/9/2022
This is the testing and logging module for churn_library.py module.
Usage:
1. Test case for all functionality.
"""

import os
import logging
import pytest
import joblib
import churn_library as cl

for directory in ["logs", "images/eda", "images/results", "./models"]:
    if not os.path.exists(directory):
        os.makedirs(directory)

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(name='raw_data')
def raw_data():
    """
    raw dataframe fixture - returns the raw dataframe from initial dataset file
    """
    try:
        raw_dataframe = cl.import_data(
            "data/bank_data.csv")
        logging.info("Raw dataframe fixture creation: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Raw dataframe fixture creation: File not found")
        raise err
    return raw_dataframe


@pytest.fixture(name='encoder_helper')
def encoder_helper(raw_data):
    """
    encoder fixture - returns the encoded dataframe on some specific column
    """
    try:
        category_lst = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"]
        df_encoded = cl.encoder_helper(raw_data, category_lst)
        logging.info("Encoded dataframe fixture creation: SUCCESS")
    except KeyError as err:
        logging.error(
            "Encoded dataframe fixture creation: Not existent column to encode")
        raise err
    return df_encoded


@pytest.fixture(name='perform_feature_engineering')
def perform_feature_engineering(encoder_helper):
    """
    feature engineering fixtures - returns 4 series containing features sequences
    """
    try:
        x_train, x_test, y_train, y_test = cl.perform_feature_engineering(
            encoder_helper)

        logging.info("Feature engineering fixture creation: SUCCESS")
    except BaseException:
        logging.error(
            "Feature engineering fixture creation: Sequences length mismatch")
        raise
    return x_train, x_test, y_train, y_test


def test_import(raw_data):
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    try:
        assert raw_data.shape[0] > 0
        assert raw_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(raw_data):
    """
    test perform eda function
    """
    cl.perform_eda(raw_data)
    images_name = ["Churn",
                   "Customer_Age",
                   "Marital_Status",
                   "Total_Trans",
                   "Heatmap"]
    for image_name in images_name:
        try:
            with open(f"images/eda/{image_name}.jpg", 'r'):
                logging.info(f"{image_name} Testing perform_eda: SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                f"{image_name} Testing perform_eda: generated images missing")
            raise err


def test_encoder_helper(encoder_helper):
    """
        test encoder helper
        """
    try:
        assert encoder_helper.shape[0] > 0
        assert encoder_helper.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't appear to have rows and columns")
        raise err
    try:
        column_names = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"]
        for column in column_names:
            assert column in encoder_helper
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe doesn't have the right encoded columns")
        raise err
    logging.info("Testing encoder_helper: SUCCESS")


def test_perform_feature_engineering(perform_feature_engineering):
    """
        test perform_feature_engineering
        """
    try:
        x_train, x_test = perform_feature_engineering[0], perform_feature_engineering[1]
        y_train, y_test = perform_feature_engineering[2], perform_feature_engineering[3]

        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
        logging.info("Testing feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing feature_engineering: Sequences length mismatch")
        raise err


def test_train_models(perform_feature_engineering):
    """
        test train_models
        """
    x_train, x_test = perform_feature_engineering[0], perform_feature_engineering[1]
    y_train, y_test = perform_feature_engineering[2], perform_feature_engineering[3]
    cl.train_models(x_train, x_test, y_train, y_test)
    try:

        joblib.load('models/rfc_model.pkl')
        joblib.load('models/logistic_model.pkl')

        logging.info("Testing testing_models: SUCCESS")

    except FileNotFoundError as err:
        logging.error("Testing train_models: The files waeren't found")
        raise err
    image_names = [
        "Logistic_Regression_Train",
        "Random_Forest_Test",
        "Logistic_Regression_Train",
        "Random_Forest_Test",
        "Feature_Importance"]
    for image_name in image_names:
        try:
            if "Train" in image_name or "Test" in image_name:
                with open("images/results/%s.png" % image_name, 'r'):
                    logging.info(
                        f"{image_name} Testing testing_models (report generation): SUCCESS")
            else:
                with open("images/results/%s.jpg" % image_name, 'r'):
                    logging.info(
                        f"{image_name} Testing testing_models (report generation): SUCCESS")
        except FileNotFoundError as err:
            logging.error(
                f"{image_name} Testing testing_models (report generation): generated images missing")
            raise err

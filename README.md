 # Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

## Project Description
Module to identify credit card customers that are most likely to churn. The completed project includes a Python package for a machine learning project that follows coding (PEP8) and engineering best practices for implementing software (modular, documented, and tested). The package will also have the flexibility of being run interactively or from the command-line interface (CLI).

## Files and data description
The following represents the schema of directories with each file inside it:

- data
   - bank_data.csv
- images
  - eda
    - churn_hist.png
    - correlation_heatmap.png
    - customer_age_hist.png
    - marital_status_bar.png
    - total_trans_ct_dist.png
  - results
    - rfc_report_test.png
    - rfc_report_train.png
    - lrc_report_test.png
    - lrc_report_train.png
    - feature_importance.png
- logs
  - churn_library.log
- models
  - logistic_model.pkl
  - rfc_model.pkl
- churn_library.py
- churn_notebook.ipynb
- churn_script_logging_and_tests.py
- README.md

In the data folder is located the dataset provided to predict the company's churn.\
In the EDA folder is made the exploratory data analysis with histograms, bar charts, heatmaps and distributions.\
In the results forder there is the the metrics reports for the random forest classifier and logistic classifier and the feature importances.\
In the logs folder there is the churn_library.log that is created when run the unit test script churn_script_logging_and_tests.py.\
In the models folder is saved the best models for the random forest classifier and logistic classifier algorithms in pickle format.\
The churn_library.py is the script that does all the steps from the feature engineering to training the best model and saving it.\
The churn_script_logging_and_tests.py is the one used for the unit test to be run using pytest.

## Running Files
1. Create and activate Virtual Environment
```
$ python3 -m venv cust_churn
$ source cust_churn/bin/activate
```
2. Install python dependencies
```angular2html
$ pip3 install -r requirements.txt
```
3. Run testcases
```angular2html
$ pytest churn_script_logging_and_tests.py
```
4. Run module
```angular2html
$ python3 churn_library.py




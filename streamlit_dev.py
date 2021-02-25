import os
import re
import time
from collections import *
from functools import *
from glob import glob
from itertools import *
from operator import *

import gensim
import gensim.corpora as corpora
import igraph
import inltk
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import pandas as pd
import plotly.express as px
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
import spacy
import streamlit as st
import streamlit.components.v1 as components
import sweetviz as sv
from autoviz.AutoViz_Class import AutoViz_Class
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from IPython import get_ipython
from more_itertools import *
from nltk.corpus import IndianCorpusReader, stopwords, wordnet
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
# from nltk import *
from nltk.stem import WordNetLemmatizer
from pandas_profiling import ProfileReport
from PIL import Image
from streamlit_pandas_profiling import st_profile_report
from textblob import TextBlob
from transformers import pipeline
from wordcloud import STOPWORDS, WordCloud

#import SessionState
# list(glob(os.getcwd()+"/**"))


def val_count(tmp: pd.Series):
    l = tmp.value_counts(dropna=True, normalize=True) * 100
    if len(l) == 0:
        return [None] * 4
    a = (l.cumsum().round(3) - 80).abs().argmin()

    return len(l), list(l.nlargest(3).round(2).items()), (
        a + 1, l.cumsum().round(2).iloc[a]), l.cumsum().median()


def missing_zero_values_2(df, corr=True):
    num_types = [
        'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64'
    ]
    oth_types = ['bool', 'datetime64[ns]', 'object', 'category']
    df_len = len(df)
    miss_zero_df = pd.DataFrame([
        (j, k, (df[i] == 0).sum() if j in num_types else 0,
         df[i].dropna().min(), df[i].dropna().max(), list(val_count(df[i])))
        for i, j, k in zip(df.columns, df.dtypes.astype('str'),
                           df.isna().sum())
    ])

    miss_zero_df.columns = [
        'col_dtype', 'null_cnt', 'zero_cnt', 'min_val', 'max_val', 'vals'
    ]
    miss_zero_df.index = df.columns
    miss_zero_df[[
        'nunique', 'top_3_largest', 'top_80_perc_approx', 'top_50_perc_share'
    ]] = pd.DataFrame(miss_zero_df['vals'].to_list(), index=miss_zero_df.index)
    miss_zero_df.drop('vals', axis=1, inplace=True)
    miss_zero_df[' % null'] = miss_zero_df['null_cnt'] * 100 / df_len
    miss_zero_df[' % zero'] = miss_zero_df['zero_cnt'] * 100 / df_len
    miss_zero_df[' % null_zero'] = miss_zero_df[' % null'] + \
        miss_zero_df[' % zero']
    miss_zero_df = miss_zero_df.sort_values([' % null_zero', ' % null'],
                                            ascending=False)  # .round(1)
    # if corr:
    #   corr_vals=get_corr(df)
    # miss_zero_df['top_corr_vals']=[[j for j in corr_vals if i in j[0]][:3] for i in miss_zero_df.index]
    return miss_zero_df


cwd = os.getcwd()
AV = AutoViz_Class()


def run_eda(df, chosen_val='Pandas Profiling'):
    if chosen_val == 'Pandas Profiling':
        pr = ProfileReport(df, explorative=True)
        st_profile_report(pr)
    elif chosen_val == 'Sweetviz':
        st.write("opening new tab")
        rep = sv.analyze(df.select_dtypes(exclude='datetime64[ns]'))
        rep.show_html()
    else:
        #AV = AutoViz_Class()
        chart_format = 'jpg'

        dft = AV.AutoViz(filename="",
                         sep=',',
                         depVar='',
                         dfte=df,
                         header=0,
                         verbose=2,
                         lowess=False,
                         chart_format=chart_format,
                         max_rows_analyzed=150000,
                         max_cols_analyzed=30)
        st.write(dft.head())
        st.write('Autoviz')
        # st.write(os.getcwd()+f"/AutoViz_Plots/empty_string/*.{chart_format}")
        for i in list(
                glob(cwd + f"/AutoViz_Plots/empty_string/*.{chart_format}")):

            image = Image.open(i)
            st.image(image)


def run_zsc(trimmed_df, text_col, labels):
    # return pd.DataFrame(
    #     zero_shot_classifier(list(collapse(df[text_col].values)),
    #                          labels,
    #                          multi_class=True))
    trimmed_df[text_col] = trimmed_df[text_col].apply(
        lambda x: zero_shot_classifier(x, labels, multi_class=True).values())
    return trimmed_df


st.set_page_config(  # Alternate names: setup_page, page, layout
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="wide",
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title='ML-Hub',
    page_icon=None,  # String, anything supported by st.image, or None.
)

option_dict = {
    'Exploratory Data Analysis': ['Pandas Profiling', 'Autoviz', 'Sweetviz'],
    'NLP': ['Sentiment Analysis', 'LDA', 'NER'],
    'Zero-Shot Topic Classification': ['sentiment', 'labels', 'both'],
    'Time Series': ['model1', 'model2'],
    'Network Graph': ['Undirected', 'Directed', 'Bidirected'],
    'Classification Models': [
        'Logistic', 'Naive_Bayes', 'KNN', 'DecisionTree', 'RandomForest',
        'LightGBM', 'AdaBoost', 'CatBoost', 'XGBoost', 'TPOT', 'SVM'
    ],
    'Regression Models': [
        'Logistic', 'Naive_Bayes', 'KNN', 'DecisionTree', 'RandomForest',
        'LightGBM', 'AdaBoost', 'CatBoost', 'XGBoost', 'TPOT', 'SVM'
    ],
    'Image Recognition': ['Object Detection', 'Facial expression', 'OCR']
}

st.sidebar.title('ML-Hub')
option = st.sidebar.selectbox('Select a task', list(option_dict.keys()))

# st.sidebar.title('ML-Hub')
# option2 = st.sidebar.selectbox(
#     'Select a task from this',
#     option_dict.get(option))

# st.title(f"{option}-{option2}")

uploaded_file = st.file_uploader("Upload a dataset")

#st.write(uploaded_file)
#help(st.number_input)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success('Successfully loaded the file {}'.format(uploaded_file.name))
    st.subheader("Sample Data")
    st.write(df.sample(5))
    # df_dtypes = pd.DataFrame(df.dtypes).reset_index()
    # df_dtypes.columns = ['column', 'dtype']
    # st.subheader("Data Types")
    # st.write(df_dtypes)
    st.subheader("Initial Analysis")
    st.write(missing_zero_values_2((df)))

    drop_cols = st.multiselect("Drop Columns", list(df.columns))
    # st.write(type(drop_cols))
    # st.write(drop_cols)
    df = df.drop(drop_cols, axis=1, errors='ignore')

    if option == 'Exploratory Data Analysis':
        option2 = st.sidebar.radio('Select a task ', option_dict.get(option))
        dep_var = st.sidebar.selectbox('select dependent variable',
                                       [''] + list(df.columns))

    elif option == 'Zero-Shot Topic Classification':

        labels = st.text_input('input labels', 'positive,negative')
        labels = labels.split(",")
        text_col = st.selectbox('select text column', list(df.columns))
        num_comments = st.slider('number of samples', 1, 1000, step=10)
        st.write(text_col)
    elif option == 'Time Series':
        option2 = st.sidebar.multiselect('Select forecasting models',
                                         option_dict.get(option))
        ts_cols = st.multiselect('select date,value columns', list(df.columns))

    elif option in ['Classification Models', 'Regression Models']:
        option2 = st.sidebar.multiselect(f'Select {option}',
                                         option_dict.get(option))
        # st.sidebar.info(
        #    "if one model is chosen, dummy model is used for comparision")
        split_type = st.sidebar.radio('split type', ['Random', 'Ordered'])
        if split_type == 'Ordered':
            order_cols = st.sidebar.multiselect('order by columns',
                                                list(df.columns))

        splits_val = st.sidebar.slider("train val test split", 0, 100,
                                       (60, 80), 5)
        split_dict_vals = {
            'train': splits_val[0],
            'val': splits_val[1] - splits_val[0],
            'test': 100 - splits_val[1]
        }
        st.sidebar.write(split_dict_vals)

        parameters = dict(zip(option2, [10] * len(option2)))
        for k, v in parameters.items():
            parameters[k] = st.sidebar.number_input(
                f"number of iterations - {k} ", 10, 1000, v, step=10, key=k)

        st.write(parameters)

    # elif option == 'Classification Models':
    #     option2 = st.sidebar.multiselect('Select Classification models',
    #                                      option_dict.get(option))
    #     # st.sidebar.info(
    #     #    "if one model is chosen, dummy model is used for comparision")
    #     parameters = dict(zip(option2, [None] * len(option2)))
    #     for k, v in parameters.items():
    #         parameters[k] = st.sidebar.number_input(
    #             f"number of iterations -{k} ", 10, 1000, step=10, key=k)
    #
    #     splits_val = st.sidebar.slider("train val test split", 0, 100,
    #                                    (60, 80), 5)
    #     split_dict_vals = {
    #         'train': splits_val[0],
    #         'val': splits_val[1] - splits_val[0],
    #         'test': 100 - splits_val[1]
    #     }
    #     st.sidebar.write(split_dict_vals)
    #     st.write(parameters)
    #
    # elif option == 'Regression Models':
    #     option2 = st.sidebar.multiselect('Select Regression models',
    #                                      option_dict.get(option))
    #     # st.sidebar.info(
    #     #    "if one model is chosen, dummy model is used for comparision")
    #     parameters = dict(zip(option2, [None] * len(option2)))
    #     for k, v in parameters.items():
    #         parameters[k] = st.sidebar.number_input(
    #             f"number of iterations -{k} ", 10, 1000, step=10, key=k)
    #     st.write(parameters)
    elif option == 'Network Graph':
        option2 = st.sidebar.selectbox('Select Network Type',
                                       option_dict.get(option))
        option3 = st.sidebar.selectbox('Select weighted Type',
                                       ['unweighted', 'weighted'])

    left_button, right_button = st.beta_columns(2)

    pressed = left_button.button('Run {}'.format(option), key='1')
    exit = right_button.button('Exit', key='2')
    if exit:
        st.write("Exiting")
        pass

    if pressed:
        st.write(option)
        with st.spinner('Running the {} '.format(option)):
            # run pandas profiler

            # st.dataframe(missing_zero_values_2(df))

            #st.write("profile report ")

            if option == 'Exploratory Data Analysis':

                run_eda(df, option2)

                st.write(dep_var)
            elif option == 'Zero-Shot Topic Classification':
                #label_cnt=int(st.text_input('input lables',1))
                #label_dict=dict(zip(['label_'+str(i+1) for i in range(label_cnt)],['']*label_cnt))
                # for k,v in label_dict.items():
                #     label=st.text_input(k,v)
                #     st.write(label)

                trimmed_df = df.sample(n=num_comments).dropna(
                    subset=[text_col])

                zero_shot_classifier = pipeline(
                    "zero-shot-classification",
                    tokenizer='/Users/apple/Desktop/Projects/Models/model_bart'
                )
                st.write(labels)
                zsc_df = run_zsc(trimmed_df, text_col, labels)
                st.write(zsc_df.head())
                # pressed_3=st.button('Submit',key='3')
                # if pressed_3:
                #     result =label_vals.title()
                #     st.success(result)
                zsc_df.to_csv('/Users/apple/Desktop/zsc_df.csv')
                st.write('noted')
            elif option == 'Classification Models':
                st.markdown("### Chosen models")
                for i in option2:

                    st.write(i)
            else:
                pass

#help(st.number_input)

st.success("Done")

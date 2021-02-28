import base64
import gc
import os
import re
import time
from collections import *
from datetime import datetime, time
from functools import *
from glob import glob
from itertools import *
from operator import *

import dask.dataframe as dd
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
import vaex
from autoviz.AutoViz_Class import AutoViz_Class
from catboost import CatBoostClassifier, CatBoostRegressor
from dataprep.eda import plot, plot_missing
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from IPython import get_ipython
from lightgbm import LGBMClassifier, LGBMRegressor
from more_itertools import *
from multipledispatch import dispatch
from nltk.corpus import IndianCorpusReader, stopwords, wordnet
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from pandas_profiling import ProfileReport
from pandas_summary import DataFrameSummary
from PIL import Image
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, Normalizer,
                                   OneHotEncoder, PowerTransformer,
                                   QuantileTransformer, StandardScaler)
from streamlit_pandas_profiling import st_profile_report
from textblob import TextBlob
from transformers import pipeline
from wordcloud import STOPWORDS, WordCloud
from xgboost import XGBClassifier, XGBRegressor

#import SessionState
# list(glob(os.getcwd()+"/**"))
cwd = os.getcwd()
#st.set_option('server.maxUploadSize', 1024)


def val_count(tmp: pd.Series):
    l = tmp.value_counts(dropna=True, normalize=True) * 100
    if len(l) == 0:
        return [None] * 4
    a = (l.cumsum().round(3) - 80).abs().argmin()

    return len(l), len(l) * 100 / tmp.count(), list(
        l.nlargest(3).round(2).items()), (
            a + 1, l.cumsum().round(2).iloc[a]), l.cumsum().median()


def set_num_type(i):
    if int(i) == i:
        return int(i)
    else:
        return i


#@dispatch(list)
def to_tuples(l):
    if len(l) <= 1:
        return []
    if len(l) == 2:
        return [tuple(l)]
    return list(zip(l, l[1:] + [l[0]]))


# @dispatch(np.ndarray)
# def to_tuples(l):
#     if len(l) <= 1:
#         return []
#     if len(l) == 2:
#         return [tuple(l)]
#     return list(zip(l, np.roll(l, -1)))


def unique_list(seq):
    seq = collapse(seq)
    seen = set([None])
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]


def missing_zero_values_2(df, corr=True):

    num_types = [
        'float16', 'float32', 'float64', 'int8', 'int16', 'int32', 'int64'
    ]
    oth_types = ['bool', 'datetime64[ns]', 'object', 'category']
    df_len = len(df)
    miss_zero_df = pd.DataFrame([
        (j, k, (df[i] == 0).sum() if j in num_types else 0,
         df[i].dropna().min(), df[i].dropna().max(), df[i].count(),
         list(val_count(df[i])))
        for i, j, k in zip(df.columns, df.dtypes.astype('str'),
                           df.isna().sum())
    ])

    miss_zero_df.columns = [
        'col_dtype', 'null_cnt', 'zero_cnt', 'min_val', 'max_val', 'count',
        'vals'
    ]
    miss_zero_df.index = df.columns
    miss_zero_df[[
        'nunique', 'uniq_perc', 'top_3_largest', 'top_80_perc_approx',
        'top_50_perc_share'
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


def get_table_download_link(df):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """

    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode(
    )  # some strings <-> bytes conversions necessary here
    href = f'#### <a href="data:file/csv;base64,{b64}">Download csv file</a>'
    return href


def export_csv(data, filename="data.csv"):
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(index=False)
    b64 = base64.b64encode(data.encode()).decode()
    return f'<a href="data:file/text;base64,{b64}" download="{filename}">Click here to download CSV</a>'


def data_distribution(p, split_dict, sort_dict={}):
    df = p.copy()
    if len(sort_dict) == 0:
        df = df.sample(frac=1)
    else:
        df = df.sort_values(list(sort_dict.keys()),
                            ascending=list(sort_dict.values()))
    # df_len = len(df)
    # df['group_dist'] = 'train'
    # df.iloc[-int(df_len * control_perc / 100):,
    #         df.columns.get_loc('group_dist')] = 'control'
    # split_dict.pop('control')
    # df['TG_offer'] = None
    l1 = list(
        map(lambda x: int(x * len(df) / 100), accumulate(split_dict.values())))
    k = np.split(df, l1[:-1])
    # print(df_len, [len(i) for i in k], len(df[df.TG_group == 'control']))
    # for i, j in enumerate(split_dict.keys()):
    #     k[i]['group_dist'] = j
    return k


def run_eda(df, dep_var="", chosen_val='Pandas Profiling'):
    if chosen_val == 'Pandas Profiling':
        pr = ProfileReport(df, explorative=True)
        st_profile_report(pr)
    elif chosen_val == 'Sweetviz':
        st.write("opening new tab")
        rep = sv.analyze(df.select_dtypes(exclude='datetime64[ns]'),
                         target_feat=dep_var)
        rep.show_html()
    elif chosen_val == 'Autoviz':
        AV = AutoViz_Class()
        chart_format = 'jpg'

        dft = AV.AutoViz(
            filename="",
            sep=',',
            depVar=dep_var,
            dfte=df,
            header=0,
            verbose=2,
            lowess=False,
            chart_format=chart_format,
            max_rows_analyzed=len(df),  #150000,
            max_cols_analyzed=df.shape[1])  #30
        st.write(dft.head())
        st.write('Autoviz')
        # st.write(os.getcwd()+f"/AutoViz_Plots/empty_string/*.{chart_format}")
        if dep_var != '':
            stored_folder = dep_var
        else:
            stored_folder = 'empty_string'
        for i in list(
                glob(cwd +
                     f"/AutoViz_Plots/{stored_folder}/*.{chart_format}")):

            st.image(Image.open(i))
    # else:
    #     st.table(DataFrameSummary(df))


def run_zsc(trimmed_df, text_col, labels):
    # return pd.DataFrame(
    #     zero_shot_classifier(list(collapse(df[text_col].values)),
    #                          labels,
    #                          multi_class=True))
    trimmed_df[text_col + '_zsc'] = trimmed_df[text_col].apply(lambda x: list(
        zero_shot_classifier(x, labels, multi_class=True).values())[-2:])
    return trimmed_df


st.set_page_config(  # Alternate names: setup_page, page, layout
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="wide",
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title='ML-hub',
    page_icon=None,  # String, anything supported by st.image, or None.
)

option_dict = {
    'Exploratory Data Analysis':
    ['Pandas Profiling', 'Autoviz', 'Sweetviz', 'Summary Table'],
    'NLP': ['Sentiment Analysis', 'LDA', 'QDA', 'NER'],
    'Zero-Shot Topic Classification': ['sentiment', 'labels', 'both'],
    'Time Series': ['model1', 'model2'],
    'Network Graph': ['Columnar', 'Chain'],
    'Clustering': ['KMeans', 'KModes', 'DBSCAN', 'AgglomerativeClustering'],
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

# lgbm_dict={
#  'learning_rate': 0.1,
#  'max_depth': -1,
#  'n_estimators': 100,
#  'n_jobs': -1,
#  'num_leaves': 31,
#  'reg_alpha': 0.0,
#  'reg_lambda': 0.0,}
#  xgb_dict={}

model_param_map = {
    'CatBoost':
    ['iterations', 'max_depth', 'num_leaves', 'learning_rate', 'reg_lambda'],
    'LightGBM': [
        'n_estimators', 'max_depth', 'n_jobs', 'learning_rate', 'num_leaves',
        'reg_alpha', 'reg_lambda'
    ],
    'XGBoost': [
        'n_estimators', 'max_depth', 'n_jobs', 'learning_rate', 'reg_alpha',
        'reg_lambda'
    ],
    'RandomForest': ['n_estimators', 'max_depth', 'n_jobs', 'max_features'],
    'Logistic': ['max_iter', 'n_jobs', 'C']
}

model_map_class = {
    'CatBoost': CatBoostClassifier,
    'LightGBM': LGBMClassifier,
    'XGBoost': XGBClassifier,
    'RandomForest': RandomForestClassifier,
    'Logistic': LogisticRegression
}
model_map_reg = {
    'CatBoost': CatBoostRegressor,
    'LightGBM': LGBMRegressor,
    'XGBoost': XGBRegressor,
    'RandomForest': RandomForestRegressor
}

st.sidebar.title('ML-Hub')
option = st.sidebar.selectbox('Select a task', list(option_dict.keys()))

# st.sidebar.title('ML-Hub')
# option2 = st.sidebar.selectbox(
#     'Select a task from this',
#     option_dict.get(option))

# st.title(f"{option}-{option2}")
st.title('Project Poseidon')
uploaded_file = st.file_uploader("Upload a dataset")

#st.write(uploaded_file)
#help(st.number_input)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    #df = vaex.read_csv(uploaded_file)
    st.success('Successfully loaded the file {}'.format(uploaded_file.name))
    st.subheader("Sample Data")
    st.write(df.sample(5))
    st.write(df.shape)

    # df_dtypes = pd.DataFrame(df.dtypes).reset_index()
    # df_dtypes.columns = ['column', 'dtype']
    # st.subheader("Data Types")
    # st.write(df_dtypes)

    st.subheader("Initial Analysis")

    if st.checkbox('Run Initial Analysis'):
        init_df = missing_zero_values_2(df)
        st.dataframe(init_df)
        st.markdown(get_table_download_link(init_df), unsafe_allow_html=True)
    drop_cols = st.multiselect("Drop Columns", list(df.columns))
    # st.write(type(drop_cols))
    # st.write(drop_cols)
    df = df.drop(drop_cols, axis=1, errors='ignore')
    columns = list(df.columns)

    if option == 'Exploratory Data Analysis':
        option2 = st.sidebar.radio('Select a task ', option_dict.get(option))
        dep_var = st.sidebar.selectbox('select dependent variable',
                                       [""] + columns)

    elif option == 'Zero-Shot Topic Classification':

        labels = st.text_input('input labels', 'positive,negative')
        labels = labels.split(",")
        text_col = st.selectbox('select text column', columns)
        num_comments = st.slider('number of samples', 1, 1000, step=10)
        st.write(text_col)
    elif option == 'Time Series':
        option2 = st.sidebar.multiselect('Select forecasting models',
                                         option_dict.get(option))
        ts_cols = st.multiselect('select date,value columns', columns)

    elif option in ['Classification Models', 'Regression Models']:
        option2 = st.sidebar.multiselect(f'Select {option}',
                                         option_dict.get(option))
        st.sidebar.info(
            "If only one model is chosen, dummy model is used for comparision")
        y_label = st.sidebar.selectbox("Select Dependant Variable", columns)
        split_type = st.sidebar.radio('split type', ['Random', 'Ordered'])
        order_map = dict({})
        if split_type == 'Ordered':
            order_cols = st.sidebar.multiselect('order by columns', columns)
            order_map = dict(
                zip(order_cols, [
                    st.sidebar.selectbox(f'{i} - ascending', [True, False],
                                         key=i) for i in order_cols
                ]))

            st.write(order_map)

        splits_val = st.sidebar.slider("train val test split", 0, 100,
                                       (60, 80), 5)
        split_dict_vals = {
            'train': splits_val[0],
            'val': splits_val[1] - splits_val[0],
            'test': 100 - splits_val[1]
        }
        st.sidebar.write(split_dict_vals)

        #parameters = dict(zip(option2, [10] * len(option2)))
        # for k, v in parameters.items():
        #     parameters[k] = st.sidebar.number_input(
        #         f" {k} -{v}", 10, 1000, v, step=10, key=k)

        parameters = [(i, model_param_map[i]) for i in option2]
        models = []
        #st.write(parameters)
        for k, v in parameters:
            st.sidebar.subheader(k)
            st.subheader(k)
            tmp = dict(zip(v, [0] * len(v)))

            mod_a = st.beta_columns(len(v))
            for j in range(len(v)):
                tmp[v[j]] = set_num_type(mod_a[j].number_input(
                    f" {k} -{v[j]}",
                    0.0,
                    1000.0,
                    value=1.0,
                    step=0.01,
                    key=f'{k}-{v[j]}'))

            # for j in v:
            #     tmp[j] = set_num_type(
            #         st.sidebar.number_input(f" {k} -{j}",
            #                                 0.0,
            #                                 1000.0,
            #                                 value=1.0,
            #                                 step=0.01,
            #                                 key=f'{k}-{j}'))

            if option == 'Classification Models':
                models.append(model_map_class[k](**tmp))
            if option == 'Regression Models':
                models.append(model_map_reg[k](**tmp))

        st.write(models)

        # try:
        #     st.write(models[0].get_params())
        # except:
        #     pass

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

        if option2 == 'Columnar':
            directed, weighted = st.sidebar.beta_columns(2)
            directed = directed.checkbox('Is Directed')
            weighted = weighted.checkbox('Is Weighted')
            st.markdown("## Nodes")
            src, dest = st.beta_columns(2)
            src = src.selectbox('select Source column', columns)
            dest = dest.selectbox('select Destination column',
                                  list(df.columns))
            if weighted:
                st.markdown('## Weights')
                src_wt, dest_wt, edge_wt = st.beta_columns(3)
                src_wt = src_wt.selectbox('select Source weight column',
                                          [None] + columns)
                dest_wt = dest_wt.selectbox('select Destination weight column',
                                            [None] + columns)
                edge_wt = edge_wt.selectbox('select Edge weight column',
                                            [None] + columns)
        elif option2 == 'Chain':
            st.markdown("## Chain")
            node_col = st.selectbox('select Chain column', columns)

    left_button, right_button = st.beta_columns(2)

    pressed = left_button.button('Run {}'.format(option), key='1')
    exit = right_button.button('Exit', key='2')
    if exit:
        st.write("Exiting")
        pass

    if pressed:
        st.write(option)

        start = datetime.now()
        with st.spinner('Running the {} '.format(option)):

            if option == 'Exploratory Data Analysis':

                run_eda(df, dep_var, option2)

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
                obj_cols = df.select_dtypes('object').columns
                model_data = pd.get_dummies(
                    df,
                    columns=[i for i in obj_cols if i != y_label]).fillna(0)
                train, val, test = data_distribution(model_data,
                                                     split_dict_vals,
                                                     order_map)
                trained_models = []
                for i in models:
                    #st.subheader(i)
                    st.dataframe(train.head())
                    le = LabelEncoder()
                    st.write(train.dtypes.value_counts())
                    i.fit(train.drop(y_label, axis=1),
                          le.fit_transform(train[y_label].astype(str)))
                    st.write('val')
                    st.write(
                        i.score(val.drop(y_label, axis=1),
                                le.transform(val[y_label].astype(str))))
                    st.write('test')
                    st.write(
                        i.score(test.drop(y_label, axis=1),
                                le.transform(test[y_label].astype(str))))

                    trained_models.append(i)
            elif option == 'Regression Models':
                st.markdown("### Chosen models")
                obj_cols = df.select_dtypes('object').columns
                model_data = pd.get_dummies(
                    df,
                    columns=[i for i in obj_cols if i != y_label]).fillna(0)
                train, val, test = data_distribution(model_data,
                                                     split_dict_vals,
                                                     order_map)
                trained_models = []
                for i in models:
                    #st.subheader(i)
                    st.dataframe(train.head())
                    st.write(train.dtypes.value_counts())
                    i.fit(train.drop(y_label, axis=1), train[y_label])
                    st.write('val')
                    st.write(i.score(val.drop(y_label, axis=1), val[y_label]))
                    st.write('test')
                    st.write(i.score(test.drop(y_label, axis=1),
                                     test[y_label]))

                    trained_models.append(i)
            elif option == 'Network Graph':

                if option2 == 'Chain':
                    #st.write(df[node_col].map(to_tuples))
                    S = igraph.Graph()
                    chains = df[node_col].apply(lambda x: x.replace('[', '').
                                                replace(']', '').split(','))
                    vertices = unique_list(chains)
                    edges = list(
                        unique_everseen(chain.from_iterable(
                            map(to_tuples, chains)),
                                        key=frozenset))

                    S.add_vertices(vertices)
                    S.add_edges(edges)

                elif option2 == 'Columnar':
                    if weighted:
                        S = igraph.Graph.DataFrame(df[[src, dest,
                                                       edge_wt]].astype(str),
                                                   directed=directed)
                    else:
                        S = igraph.Graph.DataFrame(df[[src, dest]].astype(str),
                                                   directed=directed)
                st.write(S)

                st.write(list(S.clusters()))
                # st.write(list(S.es.attributes()))
                st.write(list(S.vs))
            else:
                pass
        st.write(datetime.now() - start)

#help(st.number_input)

# a=np.random.choice(range(10),size=100)
# np.split(a,[10,30,40])
# S.vs.attributes
st.success("Done")

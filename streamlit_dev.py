import base64
import gc
import io
import os
import re
import time
from collections import *
from datetime import datetime, time
from functools import *
from glob import glob
from itertools import *
from operator import *

# import bamboolib as bam
import cv2
import dask.dataframe as dd
import gensim
import gensim.corpora as corpora
import igraph
import inltk
import matplotlib.pyplot as plt
import networkx as nx
import nltk
import numpy as np
import openpyxl
import pandas as pd
import plotly.express as px
import pyLDAvis
import pyLDAvis.gensim
import seaborn as sns
import sktime
import spacy
import streamlit as st
import streamlit.components.v1 as stc
import streamlit_theme as stt
import sweetviz as sv
import sympy
import tsboost
import vaex
from autoviz.AutoViz_Class import AutoViz_Class
from catboost import CatBoostClassifier, CatBoostRegressor
# from dataprep.eda import plot, plot_missing
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from igraph import plot as igplot
# from IPython import get_ipython
from lightgbm import LGBMClassifier, LGBMRegressor
from more_itertools import *
from multipledispatch import dispatch
from nltk.corpus import IndianCorpusReader, stopwords, wordnet
from nltk.sentiment import SentimentAnalyzer, SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from pandas_profiling import ProfileReport
from pandas_summary import DataFrameSummary
# from pandas_ui import get_df, get_meltdf, get_pivotdf
from PIL import Image
# pipreqs
from prophet import Prophet
from sklearn.ensemble import (AdaBoostClassifier, AdaBoostRegressor,
                              RandomForestClassifier, RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, Normalizer,
                                   OneHotEncoder, PowerTransformer,
                                   QuantileTransformer, StandardScaler)
from sktime.performance_metrics.forecasting import smape_loss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing as HWES
from statsmodels.tsa.stattools import adfuller, kpss
from streamlit_pandas_profiling import st_profile_report
from textblob import TextBlob
from transformers import pipeline
from wordcloud import STOPWORDS, WordCloud
from xgboost import XGBClassifier, XGBRegressor

# stt.set_theme({'primary': '#1b3388'})

# import SessionState
# list(glob(os.getcwd()+"/**"))
cwd = os.getcwd()
# st.set_option('server.maxUploadSize', 1024)

# st.title('Project Poseidon')

st.set_page_config(  # Alternate names: setup_page, page, layout
    # Can be "centered" or "wide". In the future also "dashboard", etc.
    layout="wide",
    initial_sidebar_state="auto",  # Can be "auto", "expanded", "collapsed"
    # String or None. Strings get appended with "• Streamlit".
    page_title='ML-hub',
    page_icon=None,  # String, anything supported by st.image, or None.
)
st.write(
    '<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

st.markdown('# Project Poseidon')

st.header("Welcome")


option_dict = {
    'Data Exploration': ['Lineplot', 'Barplot', 'Histogram', 'Piechart', 'Boxplot', 'Scatterplot',  'Calplot', 'Contour', 'Violin', 'map', 'Sankey', 'Jointplot', 'Pairplot', 'kde'],
    'Exploratory Data Analysis':
    ['Pandas Profiling', 'Autoviz', 'Sweetviz', 'Summary Table'],
    'NLP': ['Sentiment Analysis', 'LDA', 'QDA', 'NER'],
    'Zero-Shot Topic Classification': ['sentiment', 'labels', 'both'],
    'Time Series': ['naive', 'average', 'ARIMA', 'Exponential Smoothing', 'FFT', 'Prophet', 'Theta Forecaster', 'RegressionForecaster'],
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
    'Image Recognition':
    ['Play With Image', 'Object Detection', 'Facial expression', 'OCR'],
    'Custom Functions':
    ['func1', 'func2']
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


ts_metrics = ['mape', 'wmape', 'me', 'mae',
              'mpe', 'rmse', 'corr', 'smape_loss']


def forecast_performance(forecast, actual):
    mape = np.mean(np.abs(forecast - actual) / np.abs(actual)) * 100  # MAPE
    wmape = sum(np.abs(forecast - actual)) / sum(np.abs(actual)) * 100  # wmape
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual) / actual) * 100   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    corr = np.corrcoef(forecast, actual)[0, 1]   # corr
#     mins = np.amin(np.hstack([forecast[:,None],
#                               actual[:,None]]), axis=1)
#     maxs = np.amax(np.hstack([forecast[:,None],
#                               actual[:,None]]), axis=1)
    smape_loss_val = smape_loss(pd.Series(actual), pd.Series(forecast)) * 100
    return({'mape': mape, 'wmape': wmape, 'me': me, 'mae': mae,
            'mpe': mpe, 'rmse': rmse, 'corr': corr, 'smape_loss': smape_loss_val})


def naive_model(df, fh, seasonality):
    lag = st.sidebar.number_input('lag', 1, 10)
    return df[-lag - fh:-lag].values


def average_model(df,  fh, seasonality):
    period = st.sidebar.number_input('period', 3, 15)
    # .reset_index(drop=True)
    return df.rolling(period).mean().iloc[-1 - fh:-1].values


def hwes_model(df, fh, seasonality):
    model = HWES(df.iloc[:-fh], seasonal_periods=seasonality,
                 trend='add', seasonal='add')
    fitted = model.fit()

    return fitted.forecast(steps=fh).values


def fft_model(df, fh, seasonality):
    x = df.iloc[:-fh].values
    n = x.size
    # number of harmonics in model
    n_harm = st.sidebar.number_input('fft harmonics', 5, 100)
    t = np.arange(0, n)
    p = np.polyfit(t, x, 1)         # find linear trend in x
    x_notrend = x - p[0] * t        # detrended x
    x_freqdom = np.fft.fft(x_notrend)  # detrended x in frequency domain
    f = np.fft.fftfreq(n)              # frequencies
    indexes = range(n)
    # sort indexes by frequency, lower -> higher
    indexes = np.array(sorted(indexes, key=lambda i: np.absolute(f[i])))

    t = np.arange(0, n + fh)
    restored_sig = np.zeros(t.size)
    for i in indexes[:1 + n_harm * 2]:
        ampli = np.absolute(x_freqdom[i]) / n   # amplitude
        phase = np.angle(x_freqdom[i])          # phase
        restored_sig += ampli * np.cos(2 * np.pi * f[i] * t + phase)
    return (restored_sig + p[0] * t)[-fh:]


def prophet_model(df, fh, seasonality):
    df = df.reset_index(drop=False)
    df.columns = ['ds', 'y']
    model = Prophet(daily_seasonality=False, yearly_seasonality=True,
                    weekly_seasonality=False, interval_width=0.95)
    model = model.add_seasonality(
        name='custom', period=seasonality, fourier_order=5, prior_scale=0.02)
    model.fit(df.iloc[:-fh])
    return model.predict(df[-fh:][['ds']])['yhat'].values


    # 'ARIMA':arima_model,
    #
ts_model_map = {'naive': naive_model, 'average': average_model,
                'Exponential Smoothing': hwes_model, 'FFT': fft_model, 'Prophet': prophet_model}
#ts_model_parameters={'naive':['lag'],'average':['period'],'ARIMA':['p','d','q'],'Exponential Smoothing':[],'Prophet':[]}


def timeseries_forecasting(df, models, fh, seasonality):
    #train = df.iloc[:-fh]
    test = df.iloc[-fh:]
    test_df = pd.DataFrame(test).reset_index()

    for i in models:
        st.write(i)
        pred = ts_model_map.get(i)(df, fh, seasonality)
        # st.write(pred)
        test_df[i] = pred
        #test_df = pd.concat([test_df, pred],axis=1, ignore_index=False)

    st.markdown("<p style='color:blue;'> Result</p>",
                unsafe_allow_html=True)
    test_df.columns = ['date', 'actual'] + models
    st.dataframe(test_df)
    return test_df


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


# @dispatch(list)
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


def highlight_abs_max(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_max = s == s.abs().max()
    return ['background-color: yellow' if v else '' for v in is_max]


def highlight_abs_min(s):
    '''
    highlight the maximum in a Series yellow.
    '''
    is_min = s == s.abs().min()
    return ['background-color: yellow' if v else '' for v in is_min]


def color_null_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = 'black' if val else 'red'
    return 'color: %s' % color
# @st.cache


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
    # print(df_len, [len(i) for i in k], len(df[df.TG_group == 'control']))
    # for i, j in enumerate(split_dict.keys()):
    #     k[i]['group_dist'] = j
    return np.split(df, l1[:-1])


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
            max_rows_analyzed=len(df),  # 150000,
            max_cols_analyzed=df.shape[1])  # 30
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
    elif chosen_val == 'Summary Table':
        get_df(df)
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


################################################################################
st.sidebar.title('ML-Hub')
option = st.sidebar.selectbox('Select a task', list(option_dict.keys()))

uploaded_file = None


read_dict = {
    'csv': pd.read_csv,
    'xlsx': pd.read_excel,
    'parquet': pd.read_parquet,
    'pickle': pd.read_pickle
}


def file_upload(txt, data_type='csv'):
    data_type = st.radio('select datatype',
                         list(read_dict.keys()))

    return st.file_uploader(txt), read_dict[data_type]


if option not in ['Image Recognition']:
    uploaded_file, read_file = file_upload("Upload a dataset")
else:
    input_type = st.radio('Input Type', ['URL', 'File', 'Live'])
    if input_type == 'File':
        input_file = st.file_uploader(
            "Upload File", type=['png', 'jpg', 'svg'])
    elif input_type == 'URL':
        input_file = st.text_input("URL")
    else:
        st.subheader("Webcam Live Feed")
        run = st.radio('to run', ['run', 'stop'])
        FRAME_WINDOW = st.image([])

        camera = cv2.VideoCapture(0)
        if run == 'run':
            mirror = st.checkbox('Mirror')

        face_cascade = cv2.CascadeClassifier(cwd + '/haarcascades/face.xml')

        # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
        # Trained XML file for detecting eyes
        eye_cascade = cv2.CascadeClassifier(cwd + '/haarcascades/eye.xml')

        while run == 'run':
            ret, img = camera.read()
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # COLOR_BGR2RGB)
            # st.write(np.size(frame))

            faces = face_cascade.detectMultiScale(frame, 1.3, 5)
            for (x, y, w, h) in faces:
                # To draw a rectangle in a face
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)
                roi_gray = frame[y:y + h, x:x + w]

                # roi_color = img[y:y + h, x:x + w]
                # st.write(np.size(roi_gray))
                # Detects eyes of different sizes in the input image
                eyes = eye_cascade.detectMultiScale(roi_gray)
                # To draw a rectangle in eyes
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(roi_gray, (ex, ey), (ex + ew, ey + eh),
                                  (0, 127, 255), 2)

                # frame[y - 25:y + h + 25,
                #       x - 20:x + w + 20] = frame[y - 25:y + h + 25,
                #                                  x - 20:x + w + 20][::-1]

            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            frame = cv2.putText(frame, 'Dinesh', org, font, fontScale, color,
                                thickness, cv2.LINE_AA)
            if mirror:
                FRAME_WINDOW.image(frame[:, ::-1])
            else:
                FRAME_WINDOW.image(frame)

        else:
            st.write('Stopped')
            camera.release()
            cv2.destroyAllWindows()
# st.write(uploaded_file)
# help(st.number_input)


def filter_df(df):
    temp = df.select_dtypes(
        include=['category', 'object', 'bool']).nunique().items()
    temp = st.multiselect('select columns to filter', [
                          i for i, j in temp if j < 10])
    s = []
    for i in temp:
        col, value = st.beta_columns(2)
        col.write(i)
        chosen_value = value.multiselect(
            'select values', df[i].unique(), key=i)
        if chosen_value:
            s = s + [(i, chosen_value)]
    for fil in s:
        df = df[df[fil[0]].isin(fil[1])]
    return df


if uploaded_file is not None:
    df = read_file(uploaded_file)
    # df = vaex.read_csv(uploaded_file)
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
        with st.beta_expander('Dataframe Description'):
            st.write(df.describe(include='all', percentiles=[
                     0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.95, 0.99]))

        st.markdown('#### Additional Analysis')
        init_df = missing_zero_values_2(df)
        st.dataframe(init_df)

        st.markdown(get_table_download_link(init_df), unsafe_allow_html=True)

    drop_cols = st.multiselect("Drop Columns", list(df.columns))
    # st.write(type(drop_cols))
    # st.write(drop_cols)

    df = df.drop(drop_cols, axis=1, errors='ignore')

    if st.checkbox('Filter Data'):
        df = filter_df(df)

    columns = list(df.columns)

    if option == 'Data Exploration':
        option2 = st.sidebar.selectbox(
            'Select a task ', option_dict.get(option))

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

        # metrics
        st.sidebar.markdown('## Metrics')
        metrics = st.sidebar.multiselect('metrics', ts_metrics, default=[
                                         'mape', 'wmape', 'smape_loss', 'rmse'])

        with st.sidebar.beta_expander('Metrics Formula'):
            st.latex(
                """mape=\dfrac{1}{n}\sum{\left|\dfrac{F_{i}-A_{i}}{A_{i}}\\right|}\cdot100""")
            st.latex(
                "wmape=\dfrac{\sum{|{F_{i}-A_{i}}|}}{\sum{|A_{i}|}}\cdot100")
            st.latex(
                "smape=\dfrac{100}{n}\sum _{i=1}^{n}{\dfrac {|F_{i}-A_{i}|}{(|A_{i}|+|F_{i}|)/2}}")
            st.latex(
                "rmse={\sqrt {\dfrac {\sum _{i=1}^{n}({F_i}-{A_i})^{2}}{n}}}")
            st.latex("me=\dfrac{\sum _{i=1}^{n}{{F_i}-{A_i}}}{n}")
            st.latex("mae=\dfrac{\sum _{i=1}^{n}{|{F_i}-{A_i}|}}{n}")
            st.latex(
                "mpe=\dfrac{1}{n}\sum{\dfrac{F_{i}-A_{i}}{A_{i}}}\cdot100")
            st.latex("""
                corr={\dfrac {\sum \limits _{i=1}^{n}(A_{i}-{\\bar {A}})
                (F_{i}-{\\bar {F}})}{\sqrt {\sum \limits _{i=1}^{n}(A_{i}-{\\bar {A}})^{2}
                \sum \limits _{i=1}^{n}(F_{i}-{\\bar {F}})^{2}}}}""")

        #metrics = st.sidebar.multiselect('metrics', ts_metrics)
        st.sidebar.markdown("## Parameters")
        date_col, value_col = st.beta_columns(2)
        date_col = date_col.selectbox('choose date col',  [
                                      i for i in columns if df[i].dtype in ['datetime64[ns]', 'object']] + [None])
        value_col = value_col.selectbox('choose value col', [
            i for i in columns if df[i].dtype not in ['datetime64[ns]', 'object']])

        if date_col:
            try:
                start_date, end_date = st.beta_columns(2)
                start_date = start_date.date_input(
                    'start date', df[date_col].astype('datetime64[D]').min()).strftime('%Y-%m-%d')
                end_date = end_date.date_input(
                    'end_date', df[date_col].astype('datetime64[D]').max()).strftime('%Y-%m-%d')
                freq = st.radio('select frequency', [
                                '1D', '1W', '1M', '1Q', '1Y'])
                custom_freq = st.text_input('custom_freq', '')
                if custom_freq != '':
                    freq = custom_freq
                # st.write(df.set_index(df[date_col].astype('datetime64[D]'))
                #          [value_col].resample(freq).sum())
                df[date_col] = pd.to_datetime(df[date_col])

                st.write(f"""
                start_date: {start_date}

                end_date: {end_date}
                """)

                st.latex("wmape=\dfrac{2}{3}")
                st.latex(
                    "wmape=\dfrac{\sum{|{A{i}-F{i}}|}}{\sum{|A{i}|}}")

                df = df[df[date_col].between(start_date, end_date)]
                df = df.set_index(df[date_col])[value_col].resample(freq).sum()

                # st.write(df)
                # https://stats.stackexchange.com/questions/30569/what-is-the-difference-between-a-stationary-test-and-a-unit-root-test/235916#235916
                test_1, test_2 = st.beta_columns(2)
                with test_1:
                    with st.beta_expander("ADF Test (Unit Root Test)"):
                        regression = st.radio(
                            'regression', ['c', 'ct', 'ctt', 'nc'])
                        result = adfuller(df.values, regression=regression)

                        st.write("<p style='color:green;'>H<sub>0</sub>: Non-Stationary</p>",
                                 unsafe_allow_html=True)
                        st.write(f'ADF Statistic: {result[0]}')
                        st.write(f'p-value: {result[1]}')
                        st.write(f'n-lags: {result[2]}')
                        st.write(f'observations: {result[3]}')
                        st.write('Critical Values:')
                        for key, value in result[4].items():
                            st.write(f"{key}: {value}")
                        if result[1] <= 0.05:
                            st.info("Hypothesis Rejected")
                        else:
                            st.warning('Test Inconclusive')
                with test_2:
                    with st.beta_expander("KPSS Test (Stationary Test)"):
                        regression = st.radio(
                            'regression', ['c', 'ct'])
                        result = kpss(df.values, regression=regression)

                        st.write("<p style='color:green;'>H<sub>0</sub>: Stationary</p>",
                                 unsafe_allow_html=True)
                        st.write(f'KPSS Statistic: {result[0]}')
                        st.write(f'p-value: {result[1]}')
                        st.write(f'n-lags: {result[2]}')
                        st.write('Critical Values:')
                        for key, value in result[3].items():
                            st.write(f"{key}: {value}")
                        if result[1] <= 0.05:
                            st.info("Hypothesis Rejected")
                        else:
                            st.warning('Test Inconclusive')

                if st.checkbox('Apply diff'):
                    lag_diff = st.number_input('select lag ', 1, 10)

                    df = df.diff(lag_diff).iloc[lag_diff:]
                    st.write(df)
                fh, seasonality = st.beta_columns(2)
                fh = fh.number_input('select forecast horizon', 1, 20)
                seasonality = seasonality.number_input(
                    'select seasonality', 4, 365)
                df = timeseries_forecasting(df, option2, fh, seasonality)

            except:
                st.error("Error")
        else:
            st.error("select valid date column")
        # st.multiselect('select date,value columns', columns)

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

        # parameters = dict(zip(option2, [10] * len(option2)))
        # for k, v in parameters.items():
        #     parameters[k] = st.sidebar.number_input(
        #         f" {k} -{v}", 10, 1000, v, step=10, key=k)

        parameters = [(i, model_param_map[i]) for i in option2]
        models = []
        # st.write(parameters)
        for k, v in parameters:
            # st.sidebar.subheader(k)
            st.subheader(k)
            tmp = dict(zip(v, [0] * len(v)))

            mod_a = st.beta_columns(len(v))
            for j in range(len(v)):
                tmp[v[j]] = set_num_type(mod_a[j].number_input(
                    f"{v[j]}",
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
        st.sidebar.subheader('Metrics')

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
    gc.collect()
    if pressed:
        st.write(option)

        start = datetime.now()
        with st.spinner('Running the {} '.format(option)):

            if option == 'Exploratory Data Analysis':

                run_eda(df, dep_var, option2)

                st.write(dep_var)
            elif option == 'Zero-Shot Topic Classification':
                # label_cnt=int(st.text_input('input lables',1))
                # label_dict=dict(zip(['label_'+str(i+1) for i in range(label_cnt)],['']*label_cnt))
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
                st.table(zsc_df.head())
                # pressed_3=st.button('Submit',key='3')
                # if pressed_3:
                #     result =label_vals.title()
                #     st.success(result)

                st.markdown(export_csv(init_df), unsafe_allow_html=True)
                # zsc_df.to_csv('/Users/apple/Desktop/zsc_df.csv')

                st.write('ZSC Done')

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
                    # st.subheader(i)
                    st.dataframe(train.head())
                    le = LabelEncoder()
                    st.write(train.dtypes.value_counts())
                    i.fit(train.drop(y_label, axis=1),
                          le.fit_transform(train[y_label].astype(str)))
                    st.subheader('val')
                    if len(val) > 0:
                        st.write(
                            i.score(val.drop(y_label, axis=1),
                                    le.transform(val[y_label].astype(str))))
                    else:
                        st.write('No Data')
                    st.subheader('test')
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
                    # st.subheader(i)
                    st.dataframe(train.head())
                    st.write(train.dtypes.value_counts())
                    i.fit(train.drop(y_label, axis=1), train[y_label])
                    st.subheader('val')
                    if len(val) > 0:
                        st.write(
                            i.score(val.drop(y_label, axis=1), val[y_label]))
                    else:
                        st.write('No Data')
                    st.subheader('test')
                    st.write(i.score(test.drop(y_label, axis=1),
                                     test[y_label]))

                    trained_models.append(i)

            elif option == 'Network Graph':
                st.markdown("### Example ")
                # https://plotly.com/python/v3/igraph-networkx-comparison/
                S = igraph.Graph(directed=True)
                S.add_vertices([1, 2, 3, 4, 5, 6, 7, 8, 10])
                S.vs['id'] = [1, 2, 3, 4, 5, 6, 7, 8]
                S.vs['label'] = [1, 2, 3, 4, 5, 6, 7, 8]
                S.add_edges([(1, 2), (2, 3), (4, 5), (1, 6)])
                # igraph.drawing.plot(S,'test.png',layout=S.layout_lgl())

                # import matplotlib.pyplot as plt
                # fig,ax=plt.subplots()
                # igraph.plot(S,target=ax)
                out_png = igraph.drawing.plot(
                    S, 'temp.png', layout=S.layout_lgl())
                out_png.save('temp.png')
                st.image('temp.png')

                if option2 == 'Chain':
                    # st.write(df[node_col].map(to_tuples))
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
                st.write(f"vertex count {S.vcount()}")
                st.write(f"edge count {S.ecount()}")
                # igraph.drawing.plot(S,'test.png')
            elif option == 'Time Series':
                # fig, ax = plt.subplots(figsize=(5, 3))
                st.pyplot(plot_acf(df['actual']))
                # st.pyplot(plot_pacf(df['actual']))
                metric_map = {}
                for i in df.columns[2:]:
                    # tmp = {i: itemgetter(*metrics)(forecast_performance(
                    #     df[i], df['actual']))}
                    # metric_map = {**metric_map, **tmp}
                    metric_map[i] = itemgetter(
                        *metrics)(forecast_performance(df[i], df['actual']))

                st.write(pd.DataFrame(metric_map, index=metrics).style.apply(
                    highlight_abs_min, axis=1))
                st.write(get_table_download_link(
                    pd.DataFrame(metric_map, index=metrics)), unsafe_allow_html=True)

            else:
                st.success("Done")
                pass

        st.write(f"""
        <p style='color:orange;'> Time Taken: </p>

        {datetime.now() - start}
        """, unsafe_allow_html=True)
        # st.balloons()

# help(st.number_input)

        # a=np.random.choice(range(10),size=100)
        # np.split(a,[10,30,40])
        # S.vs.attributes
st.success("Done completed")

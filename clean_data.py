from __future__ import division
from datetime import datetime, date, time, timedelta
from dateutil.parser import parse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import re
import time
import os


def load_csv(filepath):
    # filepath = os.path.join('data', filename)
    df = pd.read_csv(filepath)

    # lowercase, replace spaces with underscores and remove underscore if at beginning of str
    s = lambda x: re.sub("_","",x,count=1)
    new_cols = [col.lower().strip().replace(' ', '_') for col in df.columns]
    df.columns = [s(col) for col in new_cols]

    # parse timestamps
    pt = lambda t: parse(t)
    df.timestamp = df.timestamp.apply(pt)
    # df.set_index('timestamp', inplace=True)
    return df

def fil_empty(df):
    # df['new_ts'] = 0
    date = '16-Jan-2017'
    frames = []

    df2 = df.set_index('timestamp')
    freq_dict = df2.groupby([df2.index.hour, df2.index.minute, df2.index.second]).count()['accx'].to_dict()
    del df2

    for dt in pd.date_range(df.timestamp.min(),df.timestamp.max(), freq='s'):
        time_tup = (dt.hour, dt.minute, dt.second)
        if time_tup in freq_dict:
            freq = freq_dict[time_tup]
            temp_df = df[df.timestamp == dt]
            tdeltas = pd.Series( [timedelta(microseconds = (1 / freq * 1000000) * n) for n in range(freq)])

            temp_df.reset_index(inplace=True, drop=True)
            temp_df['tdelta'] = tdeltas
            temp_df.timestamp = temp_df.timestamp + temp_df.tdelta
            temp_df.drop('tdelta', 1, inplace=True)

            frames.append(temp_df)

    result = pd.concat(frames)
    return result

def plot_ts(df, filename):
    ts = df.set_index('timestamp')

    accz = ts['accz'].resample('13L').mean()
    thresholdhigh = ts.thresholdhigh.values.mean()
    thresholdlow = ts.thresholdlow.values.mean()

    accz.plot(color= 'blue', linestyle='-', lw=.2)

    plt.axhline(thresholdlow, color='red', linestyle='-', linewidth=.5)
    plt.axhline(thresholdhigh, color='red', linestyle='-', linewidth=.5)

    filename += '.png'
    filepath = os.path.join('plots', filename)
    plt.savefig(filepath, format='png')
    plt.close()

def generate_plots(data_folder):
    # data_folder = 'data'
    len_progress = len(os.listdir(data_folder)[1:])
    bar = progressbar.ProgressBar(maxval=len_progress, \
        widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])

    bar.start()
    for idx, filename in enumerate(os.listdir(data_folder)[1:]):
        filepath = os.path.join(data_folder, filename)
        df = load_csv(filepath)
        filled_df = fil_empty(df)

        plot_name = re.sub(".csv","",filename,count=1)
        plot_ts(filled_df, plot_name)
        bar.update(idx + 1)

    bar.finish()

if __name__ == '__main__':
    # df = load_csv('16-Jan-201745618PM.csv')
    # df = fil_empty(df)
    # plot_ts(df)
    generate_plots('data')

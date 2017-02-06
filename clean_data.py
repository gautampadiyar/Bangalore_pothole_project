from __future__ import division
from datetime import datetime, date, time, timedelta
from dateutil.parser import parse
from pylab import *
import plotly.plotly as py
from plotly.graph_objs import *
from plotly import tools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
import re
import time
import os


def reset_isbump(df):
    df.loc[df['accz'] < 7.5, 'isbump'] = 1
    df.loc[df['accz'] > 7.5, 'isbump'] = 0
    df.loc[df['accz'] > 12.5, 'isbump'] = 1
    return df


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
    df = reset_isbump(df)
    df.drop(['thresholdlow', 'thresholdhigh', 'variable(xyz)'], 1, inplace=True)
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

import numpy

def smooth(x,window_len=11,window='hanning'):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)

    see also:

    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=numpy.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    # return y
    return y[int(window_len/2):-int(window_len/2)]

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

def smooth_demo(df):
    ts = df.set_index('timestamp')

    accz = ts['accz'].resample('13L').mean()
    x = accz.index.values
    accz = accz.values

    # t=linspace(-4,4,100)
    # x=sin(t)
    # xn=x+randn(len(t))*0.1
    # y=smooth(x)
    # ws=31
    #
    # subplot(211)
    # plot(ones(ws))

    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    #
    # hold(True)
    # for w in windows[1:]:
    #     eval('plot('+w+'(ws) )')
    #
    # # axis([0,30,0,1.1])
    #
    # legend(windows)
    # title("The smoothing windows")
    # subplot(212)
    plot(x, accz, lw=.2)
    # accz_s = smooth(accz,5,'flat')
    l = len(x)
    # plot(x, accz_s[:l])
    axhline(7.5, color='red', linestyle='-', linewidth=.5)
    axhline(12.5, color='red', linestyle='-', linewidth=.5)
    # plot(xn)
    for w in windows:
        plot(x, smooth(accz,5,w)[:l], lw=.2)
    # l=['original signal', 'signal with noise']
    l = ['signal with noise', 'thresholdlow', 'thresholdhigh']
    l.extend(windows)

    legend(l)
    title("Smoothing a noisy signal")
    show()

def smooth_demo_plotly(df):
    """
    Code adapted from:
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    ts = df.set_index('timestamp')

    accz = ts['accz'].resample('13L').mean()
    x = accz.index.values
    accz = accz.values

    windows=['flat', 'hanning', 'hamming', 'bartlett', 'blackman']

    trace = Scatter(
        x = x,
        y = accz,
        mode = 'lines'
    )
    data = [trace]
    for w in windows:
        l = len(x)
        trace = Scatter(
            x = x,
            y = smooth(accz,5,w)[:l],
            mode = 'lines'
        )
        data.append(trace)

    # Configure Layout
    layout = Layout(
    title='Smoothing a noisy signal',
    xaxis=dict(title= 'time'),
    yaxis=dict(title= 'Accz')
    )

    # Create the figure
    fig = Figure(data=data, layout=layout)

    py.plot(fig, filename='16Jan201745618PM',fileopt='extend',auto_open=True)

if __name__ == '__main__':
    # df = load_csv('data/16Jan201745618PM.csv')
    # df = fil_empty(df)
    # plot_ts(df)
    # generate_plots('data')
    smooth_demo(df)
    # smooth_demo_plotly(df)
    pass

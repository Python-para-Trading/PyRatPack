#!/usr/bin/python
# coding: utf-8

from itertools import product
from pandas import concat, MultiIndex, DataFrame, Series
from pandas.core.base import PandasObject
from matplotlib import cm #, colors
import squarify
from  matplotlib import pyplot as plt
from matplotlib.pyplot import style, legend
# from pandas.core.index import  MultiIndex
from numpy import log, sqrt




def add_data(data, new_df, name=None):
    """
    Añade un dataframe a un dataframe multiindices con un nombre definido

    Parameters
    ----------
    data : dataframe multiindice de Pandas
        Al cual se añade información
    new_df : Dataframe de Pandas
        Que se añadirá como columna al multiindices.
    name : string
        Nombre de la nueva columna nivel 0 que se añadirá
    Returns
    -------
    concatenate : Dataframe multiindices de Pandas
        Con lo nuevos datos añadidos.

    """
    if name is None:
        name = 'data_{}'.format(data.columns.levshape[0]+1)
    new_df.columns = MultiIndex.from_product([[name], new_df.columns])
    concatenate = concat([data,new_df], axis=1)

    return concatenate


def min_max_norm(data):
    """
    Escalado min-max de los datos de un dataframe de Pandas por filas. A cada
    se le resta el mínimo de los valores de la fila y se divide entre el
    rango entre el máximo y mínimo de la fila.

    Parameters
    ----------
    data :  Dataframe de Pandas

    Returns
    -------
    norm_data : Dataframe de Pandas con escalado min-max por filas
    """
    mini = data.min(axis=1)
    maxi = data.max(axis=1)
    rango = maxi - mini
    norm_data = (data.sub(mini, axis=0)).div(rango, axis=0).fillna(0)
    return norm_data


def to_weights(data):
    """
    Normaliza los pesos calculados, aplicando primero un escalado min-max a
    las filas, de forma que todos los valores quedan entre cero y uno,
    y posteriormente divide cada valor por la suma de valores de la fila,
    para que la suma de todos los valores sea igual a uno. Cumpliendo así los
    requisitos de los pesos para una cartera de activos.

    Parameters
    ----------
    data : Dataframe de Pandas con los pesos sin normalizar

    Returns
    -------
    weights : Dataframe de Pandas con los pesos normalizados.
    """

    data = min_max_norm(data)
    weights = data.div(data.sum(axis=1), axis=0).fillna(0)
    return weights


def check_weights(weights):
    """
    Comprueba que los pesos cumplen con los requisitos para ser usados en
    PyRatPack.
    Parameters
    ----------
    weights : Dataframe o Serie de Pandas
        Contiene los pesos a comprobar

    Returns
    -------
    msg : string
        Mensaje indicando si los pesos son correctos
    """
    check_1 = ((weights.sum(axis=1) > 1.01).sum() == 0)
    check_2 = ((weights > 1.01).sum().sum() == 0)
    check_3 = (weights < -0.01).sum().sum() == 0

    errors = ['Error: La suma de los pesos es superior a uno.',
              'Error: Uno o varios pesos son mayor de uno',
              'Error: Uno o varios pesos son negativos']
    checks = [check_1, check_2, check_3]

    if all(checks):
        msg = 'Comprobacion Ok. Pesos aceptables.'
    else:
        msg = 'Comprobacion con errores. Los pesos no son aceptables'
        for num, check in enumerate(checks):
            if not check:
                print(errors[check])
    # print(msg)
    return msg


def return_blocks(returns):

    sret = (returns.sum()[returns.sum() > 0]).mul(100).round(0)

    cmap = cm.YlGn
    mini = min(sret)
    maxi = max(sret)
    norm = cm.colors.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(value)) for value in sret]

    squarify.plot(sizes=sret, label=sret.index, alpha=.8, color=colors)
    plt.axis('off')
    plt.show()


def return_bars(returns):

    returns_per_asset = DataFrame(
        100 * returns.sum().sort_values(ascending=False),
        columns=['Retorno']).style.bar(align='mid',
                                       color=['#d65f5f', '#5fba7d'])
    return returns_per_asset

def only_positives (data):
    data_pos = data.where(data>0,0)
    return data_pos

def logreturn (data, period=1):

    logreturns = log(data/data.shift(period))
    return logreturns


def weekly_returns(data, column='Open', freq='W-FRI', dropna=False,
                       log=False):

    resampled_data = data[column].resample(freq, label='left',
                                 closed='left').first()
    if log:
        changes = logreturn(resampled_data)
    else:
        changes = resampled_data.pct_change()

    changes = changes.dropna() if dropna else changes
    return changes


def weights_to_weekly(weights, dropna=False):
    weights = weights.shift().resample('W-FRI', label='left',
                                       closed='left').first()

    weights = weights.dropna() if dropna else weights
    return weights

def portfolio_logreturn (returns):
    log_returns = log(returns.sum(axis=1, level=0) + 1)
    if isinstance(returns.index, MultiIndex):
        log_returns = log_returns.unstack(
            level=list(range(0, len(returns.index.levels) - 1)))
    return log_returns

def portfolio_returns_to_prices (returns):
    if isinstance(returns.index, MultiIndex):
        prices = returns.sum(axis=1, level=0).unstack(
            level=list(range(0, len(returns.index.levels) - 1)))

    else:
        prices = returns.sum(axis=1, level=0)
    prices = prices.add(1).cumprod()
    return prices


def to_pyratpack(data, weights, fit_weights=True, dropna=False, plot=False):

    if (not isinstance(data, DataFrame)) or (
    not isinstance(weights, DataFrame)):
        raise ValueError('Los parametros data y weights deben ser DataFrame de '
                         'Pandas')

    if weights.index.freqstr is None:
        weights = weights_to_weekly(weights, dropna=dropna)

    if fit_weights:
        weights = to_weights(weights)
    changes = weekly_returns(data, dropna=dropna)
    linear_returns = changes.mul(weights.shift())
    # log_returns = log(linear_returns + 1)


    if plot:
        plot_strategy(linear_returns, changes)
    return weights, linear_returns


def plot_strategy(returns, changes=None):
    # import analisis
    plt.style.use('ggplot')
    plot_benchmark = False if changes is None else True
    equity = 100 * returns.sum(axis=1).add(1).cumprod()

    #     estudio = df.copy()
    #     DD_bh, maxDD, maxDD_ini, maxDD_fin = analisis.DrawDown(estudio.Dif_Close[60:], info = False)
    #     DD, maxDD, maxDD_ini, maxDD_fin = analisis.DrawDown(returns.fillna(0), info = False)
    DD = (equity - equity.cummax()) / equity.cummax()
    legends = ['Strategy']

    #     fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(18, 12), gridspec_kw = {'height_ratios':[3, 1, 1]})
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(18, 12),
                                        gridspec_kw={
                                            'height_ratios': [4, 1, 4]})
    fig.suptitle('Strategy Equity', fontsize=20)

    if plot_benchmark:
        benchmark = 100 * changes.mean(axis=1).add(1).cumprod()
        DD_benchmark = (benchmark - benchmark.cummax()) / benchmark.cummax()
        legends = ['Strategy', 'Benchmark']
        fig.suptitle('Strategy vs Benchmark', fontsize=20)

    #     equity = 100 * (returns.cumsum() + 1)

    ax1.plot(equity, c='b')
    if plot_benchmark:
        ax1.plot(benchmark, c='r')
    ax1.set_title('Capital')
    ax1.legend(legends)

    ax2.plot(DD * 100, c='b', alpha=0.4)
    if plot_benchmark:
        ax2.plot(DD_benchmark * 100, c='r')
    ax2.fill_between(DD.index, 0, DD * 100, color='c', alpha=0.5)
    ax2.set_title('Drawdown')
    ax1.legend(legends)

    assets_equity = returns.add(1).cumprod().mul(100)
    assets_legend = assets_equity.iloc[-1].sort_values(
        ascending=False).index.tolist()
    ax3.plot(assets_equity[assets_legend])
    ax3.legend(assets_legend, ncol=3, loc='best')
    ax3.set_title('Assets')

    plt.show()
    return


def grid_backtests (data, stategy, params, plot=True):


    params_values = product(*params)
    bt_returns = {}
    for value in params_values:
        returns = stategy(data, *value)
        bt_returns[value] = returns
    bt_returns = concat(bt_returns, keys=bt_returns.keys()) \
        .unstack(level=0).swaplevel(axis=1).sort_index(axis=1, level=0)

    if plot:
        plot_n(data, bt_returns)

    return bt_returns



def plot_n(data, bt_returns):

    style.use('ggplot')

    bt_cumprod = portfolio_returns_to_prices(bt_returns)

    if isinstance(bt_returns.index, MultiIndex):

        date_index = bt_returns.index.levels[-1]
        legend_ncol = int(
            (bt_returns.index.shape[0] / bt_returns.index.levshape[
                -1]) * bt_returns.columns.levshape[0] // 25) + 1
    else:

        date_index = bt_returns.index
        legend_ncol = int(bt_returns.columns.levshape[0]//20) + 1

    # bt_cumprod = bt_rets.add(1).cumprod()
    indice_ordenado = bt_cumprod.iloc[-1].sort_values(
        ascending=False).index.tolist()
    data_returns = weekly_returns(data).loc[date_index]
    # data_returns = weekly_returns(data)

    bt_cumprod[indice_ordenado].mul(100).plot(figsize=(20, 12))

    data_returns.mean(axis=1).add(1).cumprod().mul(100).plot(linestyle='-',
                                              linewidth=3, c='k')

    legend(indice_ordenado + ['benchmark'], ncol=legend_ncol, loc='best')
    return


def betas(returns, benchmark_returns):
    benchmark_returns.name = 'benchmark'
    returns = concat([benchmark_returns, returns], axis=1)
    b = returns.cov()/returns[benchmark_returns.name].var()
    betas = (Series(b[benchmark_returns.name], index=list(returns)))[1:]
    betas.name = 'betas'
    if isinstance(returns.columns, MultiIndex):
        betas.index=pd.MultiIndex.from_tuples(betas.index)
    return betas


def betas_by_year(returns, data):
    get_year = lambda x: x.year
    by_year = returns.groupby(get_year)
    beta_years = by_year.apply(betas, data)
    return beta_years

def information_ratio(returns, benchmark_returns, nperiod=52):
     return_difference = returns.sub(benchmark_returns, axis=0)
     volatility = return_difference.std() * sqrt(nperiod)
     information_ratio = sqrt(nperiod) * return_difference.mean() / volatility
     # information_ratio.name = 'IR'
     return information_ratio

def IR_by_year(returns, benchmark_returns):
    get_year = lambda x: x.year
    by_year = returns.groupby(get_year)
    beta_years = by_year.apply(information_ratio, benchmark_returns)
    return beta_years

def sharpe (returns, nperiod=52, get_logreturn = True):
    if get_logreturn:
        returns = returns.portfolio_logreturn()
    sharpe = sqrt(nperiod) * returns.mean().div(returns.std())
    return sharpe


def extend_pandas():

    PandasObject.plot_strategy = plot_strategy
    PandasObject.check_weights = check_weights
    PandasObject.min_max_norm = min_max_norm
    PandasObject.logreturn = logreturn
    PandasObject.weekly_returns = weekly_returns
    PandasObject.portfolio_logreturn = portfolio_logreturn
    PandasObject.add_data = add_data
    PandasObject.only_positives = only_positives
    PandasObject.portfolio_returns_to_prices = portfolio_returns_to_prices
    PandasObject.return_bars = return_bars
    PandasObject.return_blocks = return_blocks
    PandasObject.betas = betas
    PandasObject.betas_by_year = betas_by_year
    PandasObject.information_ratio = information_ratio
    PandasObject.IR_by_year = IR_by_year
    PandasObject.sharpe = sharpe
    PandasObject.to_weights = to_weights

extend_pandas()
#!/usr/bin/python
# coding: utf-8

def add_data(data, name, new_df):
    from pandas import concat, MultiIndex
    new_df.columns = MultiIndex.from_product([[name], new_df.columns])
    return concat([data,new_df], axis=1)

def min_max_norm(data):
    mini = data.min(axis=1)
    maxi = data.max(axis=1)
    rango = maxi - mini
    return (data.sub(mini, axis=0)).div(rango, axis=0)

def to_weights(data):
    data = min_max_norm(data)
    return data.div(data.sum(axis=1), axis=0)


def check_weights(weights):
    check_1 = ((weights.sum(axis=1) > 1.01).sum() == 0)
    check_2 = ((weights > 1.01).sum().sum() == 0)
    check_3 = (weights < -0.01).sum().sum() == 0

    errors = ['Error: La suma de los pesos es superior a uno.',
              'Error: Uno o varios pesos son mayor de uno',
              'Error: Uno o varios pesos son negativos']
    checks = [check_1, check_2, check_3]

    if all(checks):
        msg = 'Comprobación Ok. Pesos aceptables.'
    else:
        msg = 'Comprobación con errores. Los pesos no son aceptables'
        for num, check in enumerate(checks):
            if not check:
                print(errors[check])
    print(msg)


# def backtest(data):
#     opens = data.Open.resample('W-FRI', label='left', closed='left').first()
#     pesos = data.Weights.shift().resample('W-FRI', label='left', closed='left').first()
#     returns = pesos.shift() * opens.pct_change()
#     returns_sem = returns.sum(axis=1)
#     equity = returns_sem.cumsum() + 1
#     dd = (equity - equity.cummax())/ equity.cummax()
#     return returns, equity, dd


def return_blocks(returns):
    from matplotlib import cm, colors
    import squarify
    from  matplotlib import pyplot as plt

    sret = round(100 * returns.sum()[returns.sum() > 0], 2)

    cmap = cm.YlGn
    mini = min(sret)
    maxi = max(sret)
    norm = colors.Normalize(vmin=mini, vmax=maxi)
    colors = [cmap(norm(value)) for value in sret]

    squarify.plot(sizes=sret, label=sret.index, alpha=.8, color=colors)
    plt.axis('off')
    plt.show()


def return_distribution(returns):
    from pandas import DataFrame
    returns_per_asset = DataFrame(
        100 * returns.sum().sort_values(ascending=False),
        columns=['Retorno']).style.bar(align='mid',
                                       color=['#d65f5f', '#5fba7d'])
    return returns_per_asset


def get_weekly_pct_change(data, dropna=False):
    changes = data.Open.resample('W-FRI', label='left',
                                 closed='left').first().pct_change()
    changes = changes.dropna() if dropna else changes
    return changes


def weights_to_weekly(weights, dropna=False):
    weights = weights.shift().resample('W-FRI', label='left',
                                       closed='left').first()

    weights = weights.dropna() if dropna else weights
    return weights


def get_weekly_returns(data, weights, fit_weights=True, dropna=False, plot=True):

    from pandas import DataFrame

    if (not isinstance(data, DataFrame)) or (
    not isinstance(weights, DataFrame)):
        raise ValueError('data y pesos deben ser DataFrame de Pandas')

    if weights.index.freqstr is None:
        weights = weights_to_weekly(weights, dropna=dropna)

    if fit_weights:
        weights = to_weights(weights)
    changes = get_weekly_pct_change(data, dropna=dropna)
    returns = changes.mul(weights.shift())

    if plot:
        plot_strategy(returns, changes)
    return weights, returns


def plot_strategy(returns, changes=None):
    import analisis
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    plot_benchmark = False if changes is None else True
    equity = 100 * returns.sum(axis=1).cumsum().add(1)

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
        benchmark = 100 * changes.mean(axis=1).cumsum().add(1)
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

    ax3.plot(returns.cumsum().add(1).mul(100))
    ax3.legend(returns.columns, ncol=2)
    ax3.set_title('Assets')

    plt.show()
    return


def optimize (data, stategy, rango, plot=True):
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')
    from pandas import concat
    bt_returns = {}
    for value in rango:
        returns = stategy(data, value)
        bt_returns[value] = returns
    bt_returns = concat(bt_returns, keys=bt_returns.keys()) \
        .unstack(level=0).swaplevel(axis=1).sort_index(axis=1, level=0)

    if plot:
        indice_ordenado = bt_returns.sum(axis=1, level=0).cumsum().iloc[
            -1].sort_values(ascending=False).index.tolist()
        (100 * bt_returns[indice_ordenado].sum(axis=1, level=0).cumsum()).plot(
            alpha=0.9, figsize=(18, 12))
        (100 * data.Open.dropna().resample('W-FRI', label='left',
                                           closed='left').first().pct_change().mean(
            axis=1).cumsum()).plot(linestyle='-.',
                                   linewidth=3, c='k')
    return bt_returns

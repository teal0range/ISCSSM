# t-1期6月到t期7月 匹配 t期到t+1期回报
import pandas as pd
from Io import *
from settings import *
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

io = CsvIo()


def preprocess():
    QX_DIVIDENDPREMIUM = io.readData("QX_DIVIDENDPREMIUM")
    QX_DIVIDENDPREMIUM['SgnYear'] = QX_DIVIDENDPREMIUM['SgnYear'].astype(str)
    QX_DIVIDENDPREMIUM = QX_DIVIDENDPREMIUM[['SgnYear', 'LogPDND']].set_index('SgnYear').sort_index()

    QX_FUNDDISCOUNTPREMIUM = io.readData("QX_FUNDDISCOUNTPREMIUM")
    QX_FUNDDISCOUNTPREMIUM['SgnYear'] = QX_FUNDDISCOUNTPREMIUM['TradingDate'].str[:4]
    QX_FUNDDISCOUNTPREMIUM['CovertRate'] = QX_FUNDDISCOUNTPREMIUM['CovertRate'].clip(
        lower=QX_FUNDDISCOUNTPREMIUM['CovertRate'].quantile(0.05),
        upper=QX_FUNDDISCOUNTPREMIUM['CovertRate'].quantile(0.95)
    )
    QX_FUNDDISCOUNTPREMIUM = QX_FUNDDISCOUNTPREMIUM.groupby("SgnYear").apply(lambda x: x['CovertRate'].mean()). \
        sort_index().rename("FUNDDISCOUNTPREMIUM")

    QX_IPO = io.readData("QX_IPO")
    QX_IPO['SgnYear'] = QX_IPO['ListedDate'].str[:4]
    QX_IPO['ReturnRate'] = QX_IPO['ReturnRate'].clip(lower=QX_IPO['ReturnRate'].quantile(0.05),
                                                     upper=QX_IPO['ReturnRate'].quantile(0.95))
    QX_IPORETURN = QX_IPO.groupby("SgnYear").apply(lambda x: x['ReturnRate'].mean()).sort_index().rename("IPORETURN")
    QX_IPONUMBER = QX_IPO.groupby("SgnYear").apply(lambda x: len(x)).sort_index().rename("IPONUMBER")

    QX_STOCKRATE = io.readData("QX_STOCKRATE")
    QX_STOCKRATE['SgnYear'] = QX_STOCKRATE['SgnYear'].astype(str)
    QX_STOCKRATE = QX_STOCKRATE.set_index('SgnYear').sort_index()

    QX_TRM = io.readData("QX_TRM")
    QX_TRM = QX_TRM[(QX_TRM['MarketType'] == MARKETTYPE)]
    TURNOVER = 'TurnoverRate' + str(1 + 1 * TRADABLE_EQUITY)
    QX_TRM[TURNOVER] = QX_TRM[TURNOVER].clip(lower=QX_TRM[TURNOVER].quantile(0.05),
                                             upper=QX_TRM[TURNOVER].quantile(0.95))
    QX_TRM['SgnYear'] = QX_TRM['TradingDate'].str[:4]
    QX_TRM = QX_TRM.groupby("SgnYear").apply(lambda x: np.log(x[TURNOVER].sum())).rename("TurnoverRate")
    return pd.concat([QX_DIVIDENDPREMIUM, QX_TRM, QX_IPONUMBER, QX_IPORETURN, QX_STOCKRATE, QX_FUNDDISCOUNTPREMIUM],
                     axis=1).sort_index().dropna()


def laggedVars():
    pre = preprocess()
    data = pd.concat([pre, pre.shift(1).rename(columns={x: "lagged_" + x for x in pre.columns.tolist()})],
                     axis=1).dropna()
    io.saveData("st_description", data.describe().reset_index())
    plots(pre)
    data.iloc[:, :] = (data.copy() - data.mean()) / data.copy().std()
    return data


def plots(data: pd.DataFrame):
    x = data.index.tolist()
    for column in data.columns.tolist():
        y = data[column].values.tolist()
        tick_spacing = 5
        plt.figure(figsize=(6, 4))
        fig, ax = plt.subplots(1, 1)
        ax.plot(x, y)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
        plt.plot(x, y, marker='v', color='grey', label=column)
        plt.legend()
        plt.savefig("pictures/{}.png".format(column))


def PrincipleComponents():
    data = io.readData("st_FullSentimentFactor").set_index('SgnYear')
    raw_mat = data.iloc[:, :].to_numpy()
    pca = PCA(n_components=5)
    pca.fit(raw_mat)
    summary = {
        '12个变量':data.columns.tolist(),
        '12变量取前5个主成分平均': np.mean(pca.components_, axis=0).tolist(),
        '前5个主成分方差解释比例': np.sum(pca.explained_variance_ratio_)}
    targetSeries = getSeries(coefficient=pca.components_)
    chooseList = []
    corrcoef = {}
    for idx, col in enumerate(data.columns.tolist()[:6]):
        corrcoef[col] = np.corrcoef(data[col], targetSeries)[0][1]
        corrcoef['lagged_' + col] = np.corrcoef(data['lagged_' + col], targetSeries)[0][1]
        if np.abs(corrcoef[col]) > np.abs(corrcoef['lagged_' + col]):
            chooseList.append(idx)
        else:
            chooseList.append(idx + 6)
    summary['12个变量与主成分的相关系数'] = corrcoef
    summary['相关系数过滤剩下的6个变量'] = np.array(data.columns.tolist())[chooseList].tolist()
    summary['相关系数过滤掉6个变量系数'] = pca.components_[0][chooseList].tolist()
    result = getSeries(pca.components_[0][chooseList].reshape(1, -1), np.array(data.columns.tolist())[chooseList])
    summary['与12变量相关系数'] = np.corrcoef(result, targetSeries)[0][1]
    with open("sentimentSummary.json", 'w') as fp:
        json.dump(summary, fp, ensure_ascii=False, indent=4)
    return pd.DataFrame(result).reset_index()


def getSeries(coefficient, subset=None):
    data = io.readData("st_FullSentimentFactor").set_index('SgnYear')
    # data = data[['FUNDDISCOUNTPREMIUM', 'lagged_TurnoverRate', 'IPONUMBER',
    #              'lagged_IPORETURN', 'SRate', 'lagged_LogPDND']]
    if subset is None:
        subset = data.columns.tolist()
    data = data[subset]
    data['SENTIMENT'] = (coefficient @ data.copy().T.to_numpy())[0]
    return data['SENTIMENT']


def run():
    io.saveData("st_FullSentimentFactor", laggedVars().reset_index().rename(columns={"index": "SgnYear"}))
    io.saveData("st_SENTIMENT", PrincipleComponents())


if __name__ == '__main__':
    run()

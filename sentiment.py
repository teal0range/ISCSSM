# t-1期6月到t期7月 匹配 t期到t+1期回报
import pandas as pd
from Io import *
from settings import *
import numpy as np
from sklearn.decomposition import PCA

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
    data.iloc[:, :] = (data.copy() - data.mean()) / data.copy().std()
    return data


def PrincipleComponents():
    data = io.readData("st_FullSentimentFactor").set_index('SgnYear')
    data = data[['FUNDDISCOUNTPREMIUM', 'lagged_TurnoverRate', 'IPONUMBER',
                 'lagged_IPORETURN', 'SRate', 'lagged_LogPDND']]
    raw_mat = data.iloc[:, :].to_numpy()
    pca = PCA(n_components=1)
    pca.fit(raw_mat)
    print(pca.components_)
    return pd.DataFrame(getSeries(coefficient=pca.components_)).reset_index()


def getSeries(coefficient, subset=None):
    data = io.readData("st_FullSentimentFactor").set_index('SgnYear')
    data = data[['FUNDDISCOUNTPREMIUM', 'lagged_TurnoverRate', 'IPONUMBER',
                 'lagged_IPORETURN', 'SRate', 'lagged_LogPDND']]
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

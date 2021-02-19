# t-1期6月到t期7月 匹配 t期到t+1期回报
import pandas as pd
from Io import *
from settings import *
import numpy as np

io = CsvIo()


def preprocess():
    QX_DIVIDENDPREMIUM = io.readData("QX_DIVIDENDPREMIUM")
    QX_DIVIDENDPREMIUM['SgnYear'] = QX_DIVIDENDPREMIUM['SgnYear'].astype(str)
    QX_DIVIDENDPREMIUM = QX_DIVIDENDPREMIUM[['SgnYear', 'LogPDND']].set_index('SgnYear').sort_index()

    QX_FUNDDISCOUNTPREMIUM = io.readData("QX_FUNDDISCOUNTPREMIUM")
    QX_FUNDDISCOUNTPREMIUM['SgnYear'] = QX_FUNDDISCOUNTPREMIUM['TradingDate'].str[:4]
    QX_FUNDDISCOUNTPREMIUM = QX_FUNDDISCOUNTPREMIUM.groupby("SgnYear").apply(lambda x: x['CovertRate'].mean()).\
        sort_index().rename("FUNDDISCOUNTPREMIUM")

    QX_IPO = io.readData("QX_IPO")
    QX_IPO['SgnYear'] = QX_IPO['ListedDate'].str[:4]
    QX_IPORETURN = QX_IPO.groupby("SgnYear").apply(lambda x: x['ReturnRate'].mean()).sort_index().rename("IPORETURN")
    QX_IPONUMBER = QX_IPO.groupby("SgnYear").apply(lambda x: len(x)).sort_index().rename("IPONUMBER")

    QX_STOCKRATE = io.readData("QX_STOCKRATE")
    QX_STOCKRATE['SgnYear'] = QX_STOCKRATE['SgnYear'].astype(str)
    QX_STOCKRATE = QX_STOCKRATE.set_index('SgnYear').sort_index()

    QX_TRM = io.readData("QX_TRM")
    QX_TRM = QX_TRM[(QX_TRM['MarketType'] == 1)]
    QX_TRM['SgnYear'] = QX_TRM['TradingDate'].str[:4]
    QX_TRM = QX_TRM.groupby("SgnYear").apply(lambda x: np.log(x['TurnoverRate' + str(1 + 1 * TRADABLE_EQUITY)].sum())).\
        rename("LogTurnoverRate")
    return pd.concat([QX_DIVIDENDPREMIUM, QX_TRM, QX_IPONUMBER, QX_IPORETURN, QX_STOCKRATE, QX_FUNDDISCOUNTPREMIUM],
                     axis=1).sort_index().dropna()


def run():
    pass


if __name__ == '__main__':
    print(preprocess())
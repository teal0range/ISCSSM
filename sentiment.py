# t-1期6月到t期7月 匹配 t期到t+1期回报
import pandas as pd
from Io import *

io = CsvIo()


def preprocess():
    QX_DIVIDENDPREMIUM = io.readData("QX_DIVIDENDPREMIUM")
    QX_DIVIDENDPREMIUM = QX_DIVIDENDPREMIUM[['SgnYear', 'LogPDND']]
    QX_FUNDDISCOUNTPREMIUM = io.readData("QX_FUNDDISCOUNTPREMIUM")
    QX_FUNDDISCOUNTPREMIUM['SgnYear'] = QX_FUNDDISCOUNTPREMIUM['TradingDate'].str[:4]
    QX_FUNDDISCOUNTPREMIUM = QX_FUNDDISCOUNTPREMIUM.groupby("SgnYear").apply(lambda x: x['CovertRate'].mean())
    QX_IPO = io.readData("QX_IPO.csv")
    QX_IPO['SgnYear'] = QX_IPO['ListedDate'].str[:4]
    QX_IPORETURN = QX_IPO.groupby("SgnYear").apply(lambda x: x['ReturnRate'].mean())
    QX_IPONUMBER = QX_IPO.groupby("SgnYear").apply(lambda x: x['ReturnRate'].count())

    QX_STOCKRATE = io.readData("QX_STOCKRATE")

    QX_TRM = io.readData("QX_TRM")
    QX_TRM = QX_TRM[(QX_TRM['MarketType'] == 1) | (QX_TRM['MarketType'] == 3)]
    QX_TRM['SgnYear'] = QX_TRM['TradingDate'].str[:4]


def run():
    pass

# t-1期6月到t期7月 匹配 t期到t+1期回报
from typing import Union

from Io import *
import datetime
import pandas as pd
import numpy as np
from settings import *

io = CsvIo()


def preprocess():
    ME = io.readData("EVA_Structure")
    ME = ME[["Symbol", "EndDate", "MarketValue"]]
    ME = ME[ME['EndDate'].str[5:7] == '06']
    ME = ME[judgeMarket(ME)]
    ME['SgnYear'] = ME['EndDate'].str[:4].astype(int)
    ME = ME[["Symbol", "SgnYear", "MarketValue"]].rename(columns={"Symbol": "Stkcd"})
    ME["MarketValue"] = ME["MarketValue"].clip(
        lower=ME["MarketValue"].quantile(0.005),
        upper=ME['MarketValue'].quantile(0.995)
    )
    ME = ME.set_index(['Stkcd', 'SgnYear'])

    Age = io.readData("IPO_Cobasic")
    Age = Age[['Stkcd', 'Estbdt']].drop_duplicates(subset='Stkcd', keep='first').dropna()
    Age = Age[judgeMarket(Age)]
    res = []
    for t in range(2000, 2021, 1):
        tmp = Age.copy()
        tmp['SgnYear'] = t
        tmp = tmp[tmp['Estbdt'] < str(t + 1)]
        tmp['Age'] = np.vectorize(judgeAge)(
            tmp['Estbdt'],
            (tmp['SgnYear'] + 1).astype(str) + "-01-01"
        )
        res.append(tmp)
    Age = pd.DataFrame().append(res)
    Age = Age[['Stkcd', 'SgnYear', 'Age']].dropna()
    Age['Age'] = Age['Age'].clip(lower=Age['Age'].quantile(0.005), upper=Age['Age'].quantile(0.995))
    Age = Age.set_index(['Stkcd', 'SgnYear'])

    Earnings = io.readData("FI_T").dropna()
    Earnings = Earnings[(Earnings['Typrep'] == 'A') & (Earnings['Accper'].str[5:7] == '12')]
    Earnings = Earnings[judgeMarket(Earnings)]
    Earnings['SgnYear'] = Earnings['Accper'].str[:4].astype(int) + 1
    Earnings['dummy_Earnings'] = np.where(Earnings['Earnings'] > 0, 1, 0)
    Earnings = Earnings[['Stkcd', 'SgnYear', 'Earnings', 'dummy_Earnings']].drop_duplicates(subset=['Stkcd', 'SgnYear'])
    Equity = io.readData("FS_Combas")
    Equity = Equity[(Equity['Typrep'] == 'A') & (Equity['Accper'].str[5:7] == '12') & (judgeMarket(Equity))]
    Equity['SgnYear'] = Equity['Accper'].str[:4].astype(int) + 1
    Equity = Equity[['Stkcd', 'SgnYear', 'Equity']].drop_duplicates(subset=['Stkcd', 'SgnYear'])
    EBE = pd.merge(Earnings, Equity, on=['Stkcd', 'SgnYear'])
    EBE['E+/BE'] = np.where(EBE['Earnings'] / EBE['Equity'] < 0, 0, EBE['Earnings'] / EBE['Equity'])
    EBE = EBE[['Stkcd', 'SgnYear', 'E+/BE', 'dummy_Earnings']]
    EBE['E+/BE'] = EBE['E+/BE'].clip(lower=EBE['E+/BE'].quantile(0.005), upper=EBE['E+/BE'].quantile(0.995))
    EBE = EBE.set_index(['Stkcd', 'SgnYear'])

    DBE = io.readData("FI_TE").fillna(0)
    DBE = DBE[(DBE['Typrep'] == 'A') &
              (DBE['Accper'].str[5:7] == '12') &
              judgeMarket(DBE)]
    DBE['SgnYear'] = DBE['Accper'].str[:4].astype(int) + 1
    DBE = DBE[["Stkcd", "SgnYear", "D+/BE"]].dropna().drop_duplicates(subset=['Stkcd', 'SgnYear'])
    DBE['D+/BE'] = DBE['D+/BE'].clip(lower=DBE['D+/BE'].quantile(0.005), upper=DBE['D+/BE'].quantile(0.995))
    DBE['dummy_Dividends'] = np.where(DBE['D+/BE'] > 0, 1, 0)
    DBE = DBE.set_index(['Stkcd', 'SgnYear'])

    PPEA = io.readData("AIQ_LCFinIndexY")
    PPEA["RDExpenses"] = PPEA["RDExpenses"].fillna(0)
    PPEA = PPEA.dropna()
    PPEA['SgnYear'] = PPEA['Accper'].str[:4].astype(int) + 1
    PPEA = PPEA[(PPEA['Accper'].str[5:7] == '12') &
                judgeMarket(PPEA)]
    PPEA['PPE/A'] = PPEA['FixedAssets'] / PPEA['TotalAssets']
    PPEA['RD/A'] = PPEA['RDExpenses'] / PPEA['TotalAssets']
    PPEA = PPEA[['Stkcd', 'SgnYear', 'PPE/A', 'RD/A']].set_index(['Stkcd', 'SgnYear'])

    BEME = pd.merge(Equity, ME, on=['Stkcd', 'SgnYear']).drop_duplicates(subset=['Stkcd', 'SgnYear'])
    BEME['BE/ME'] = BEME['Equity'] / BEME['MarketValue']
    BEME = BEME[['Stkcd', 'SgnYear', 'BE/ME']].set_index(['Stkcd', 'SgnYear'])

    GS = io.readData("EI").fillna(0)
    GS = GS[(GS['Accper'].str[5:7] == '12') &
            (GS['Typrep'] == 'A') &
            judgeMarket(GS)]
    GS['SgnYear'] = GS['Accper'].str[:4].astype(int) + 1
    GS = GS[['Stkcd', 'SgnYear', 'GS']].set_index(['Stkcd', 'SgnYear'])

    factors = pd.concat([stockReturns(), ME, Age, EBE, DBE, PPEA, BEME, GS], axis=1)
    factors = factors.dropna()
    col = [c for c in factors.columns.tolist() if "dummy not in c"]
    factors[col] = factors[col].clip(lower=factors[col].quantile(0.005), upper=factors[col].quantile(0.995), axis=1)
    io.saveData("fe_describeFactors", factors.describe().reset_index())
    factors = factors.reset_index()
    factors["tMinus1"] = factors['SgnYear'] - 1
    io.saveData("fe_factors", factors)
    return factors


def judgeMarket(ME):
    cols = 'Symbol' if 'Symbol' in ME.columns.tolist() else 'Stkcd'
    if MARKETTYPE == 3:
        return ME[cols] < 10000
    else:
        return (ME[cols] < 688000) & (ME[cols] >= 600000)


def generateLower(t):
    return datetime.datetime(year=t, month=7, day=1)


def generateUpper(t):
    return datetime.datetime(year=t + 1, month=6, day=30)


def judgeAge(date: Union[datetime.datetime, str], currentTime: Union[datetime.datetime, str]):
    date = parseDate(date)
    currentTime = parseDate(currentTime)
    return (currentTime - date).days / 365


def parseDate(date: Union[datetime.datetime, str]) -> datetime.datetime:
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d")
    return date


def parseDateStr(date: Union[datetime.datetime, str]) -> str:
    if isinstance(date, datetime.datetime):
        date = datetime.datetime.strftime(date, "%Y-%m-%d")
    return date


def stockReturns():
    sr = io.readData("raw_StockPrice").rename(columns={"fullCode": "Stkcd"})
    sr['Stkcd'] = sr['Stkcd'].str[:6].astype(int)
    sr = sr[(sr['tradeDate'].str[5:7] == '06') & judgeMarket(sr)]
    sr = sr.groupby("Stkcd").apply(
        lambda x: pd.concat([x.sort_values(by=['tradeDate'])['tradeDate'],
                             x.sort_values(by=['tradeDate'])['Close'].rolling(2).
                            apply(lambda y: np.log(y.iloc[1] / y.iloc[0]))], axis=1)). \
        dropna().reset_index(level=1, drop=True).reset_index()
    sr['SgnYear'] = sr['tradeDate'].str[:4].astype(int) - 1
    sr['Close'] = sr['Close'].clip(lower=sr['Close'].quantile(0.005), upper=sr['Close'].quantile(0.995))
    return sr[['Stkcd', 'SgnYear', 'Close']].set_index(['Stkcd', 'SgnYear']).rename(columns={"Close": "Return"})


def run():
    preprocess()
    # stockReturns()


if __name__ == '__main__':
    run()

import pandas as pd
import numpy as np
from settings import *
from Io import *
from tqdm import tqdm


class sentimentGroup:
    io = CsvIo()
    _tiles = Tiles
    _sentiment = io.readData("st_SENTIMENT")
    _factors = pd.merge(_sentiment, io.readData("fe_factors"), on=['SgnYear'])
    _cols = [c for c in _factors.columns.tolist()[4:-1] if 'dummy' not in c]

    def __init__(self, col=None):
        if col is None:
            col = 'GS'
        self.col = col
        self._factors['MV'] = self._factors['MarketValue']
        self._factors = self._factors[['SgnYear', 'Stkcd', 'Return', 'SENTIMENT', 'MV', col]].copy()
        self.stockReturn = self.io.readData("fe_monthlyReturn")
        self.stockReturn['month'] = self.stockReturn['tradeDate'].str[5:7].astype(int)

    def getFactors(self):
        for item in self._cols:
            yield item

    def fetch(self):
        self._factors['positiveSentiment'] = self._factors['SENTIMENT'] > 0
        tqdm.pandas(desc="bar")
        data = self._factors.sort_values(by=["SgnYear", 'positiveSentiment', self.col]). \
            groupby(["SgnYear", 'positiveSentiment']). \
            progress_apply(
            lambda x: pd.Series([self.getVMReturn(x.iloc[int(len(x) / Tiles * i):int(len(x) / Tiles * (i + 1))])
                                 for i in range(Tiles)])).reset_index()
        data = data.drop('SgnYear', axis=1).groupby("positiveSentiment").apply(lambda x: x.mean())
        data = data.rename(columns={x: - x - 1 for x in data.columns.tolist() if not isinstance(x, str)})
        data = data.rename(columns={x: - x for x in data.columns.tolist() if not isinstance(x, str)})
        data['10-1'] = data[10] - data[1]
        data['10-5'] = data[10] - data[5]
        data['5-1'] = data[5] - data[1]
        data = data.drop('positiveSentiment', axis=1).reset_index()
        self.io.saveData("group_{}".format(self.col).replace("/", "").replace("+", ""), data)
        return data

    def getVMReturn(self, df):
        codeList = df[['SgnYear', 'Stkcd', 'MV']].drop_duplicates()
        codeList = pd.merge(codeList, self.stockReturn, on=['SgnYear', 'Stkcd'])
        codeList = codeList.groupby("month").apply(lambda x: np.sum(x['Close'] * x['MV'] / x['MV'].sum())).mean() * 100
        return codeList


if __name__ == '__main__':
    print(sentimentGroup('MarketValue').fetch())
    for c in sentimentGroup().getFactors():
        print(sentimentGroup(c).fetch())

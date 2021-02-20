from collections import defaultdict
import collections
import pandas as pd
import statsmodels.api as sm
import json
from tqdm import tqdm
from Io import *

io = CsvIo()


class baselineModel:
    data = pd.merge(io.readData("fe_factors").drop("tMinus1", axis=1),
                    io.readData("st_SENTIMENT"), on="SgnYear")
    cols = data.columns.tolist()[3: -1]

    def __init__(self):
        self.regressionSummary = defaultdict(lambda: defaultdict(lambda: 0))
        self.regressionTResults = pd.DataFrame()
        self.regressionPResults = pd.DataFrame()
        print(self.data.columns.tolist())

    def fetch(self):
        with tqdm(total=len(self.cols)) as bar:
            for c in self.cols:
                self.getRegressionResult(c)
                bar.update()

    def getRegressionResult(self, featureName):
        regressionData = self.data[['Stkcd', 'SgnYear', 'Return', featureName, 'SENTIMENT']].copy()
        regressionData['cross'] = regressionData[featureName] * regressionData['SENTIMENT']
        regressionData = regressionData.groupby("Stkcd").apply(lambda x: None if len(x) < 8 else x)
        regressionData = regressionData.reset_index(drop=True).dropna()
        regressionData.groupby("Stkcd").apply(self.regress)
        summary = self.outputSummary()
        with open("Data/result/regressionSummary_{}.json"
                          .format(featureName.replace("/", "").replace("+", "")), 'w') as fp:
            json.dump(summary, fp, indent=4)
        self.regressionSummary = defaultdict(lambda: defaultdict(lambda: 0))

    def regress(self, regressionData):
        reg = sm.OLS(regressionData.iloc[:, 2], regressionData.iloc[:, 3:])
        res = reg.fit()
        self.regressionTResults = self.regressionTResults.append(res.tvalues, ignore_index=True)
        self.regressionPResults = self.regressionPResults.append(res.pvalues, ignore_index=True)
        for key, val in res.tvalues.to_dict().items():
            self.regressionSummary[key][baselineModel.getSig(val)] += 1

    @staticmethod
    def getSig(num):
        s = (num < 0.01) * 1 + (num < 0.05) * 1 + (num < 0.1) * 1
        return s * '*'

    def outputSummary(self):
        data = dict(self.regressionSummary)
        for key, val in data.items():
            data[key] = dict(val)
        return data


if __name__ == '__main__':
    baselineModel().fetch()

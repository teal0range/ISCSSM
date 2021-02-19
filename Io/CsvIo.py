import os

import pandas as pd

from Io.BaseIo import BaseIo


class CsvIo(BaseIo):

    def __init__(self, config=None):
        """
        以csv文件形式输出
        :param config: 配置存储路径
        """
        if config is None:
            config = {"data": "data"}
        super().__init__()
        self.dataPath = config['data']
        if not os.path.exists(os.path.join('Data', self.dataPath)):
            os.makedirs(os.path.join('Data', self.dataPath))
        self.rawPath = 'raw'
        if not os.path.exists(os.path.join('Data', self.rawPath)):
            os.makedirs(os.path.join('Data', self.rawPath))

    def saveData(self, key, df: pd.DataFrame):
        path = self.getDataPath(key)
        if os.path.exists(path):
            pass
        df.to_csv(path, index=False)

    def hasKey(self, key):
        return os.path.exists(os.path.join(self.getDataPath(key)))

    def readData(self, key) -> pd.DataFrame:
        try:
            return pd.read_csv(self.getDataPath(key), low_memory=False)
        except FileNotFoundError:
            pass

    def remove(self, key):
        os.remove(self.getDataPath(key))

    def clear(self, prefix=""):
        keys = self.getAllKeys()
        for key in keys:
            if key.startswith(prefix):
                self.remove(key)

    def readAllData(self) -> dict:
        keys = self.getAllKeys()
        return {key: self.readData(key) for key in keys}

    def getAllKeys(self):
        path = os.path.join('Data', self.dataPath)
        for root, dirs, files in os.walk(path):
            return [file[:file.rfind(".csv")] for file in files if file.endswith(".csv")]

    def getDataPath(self, key):
        return os.path.join('Data', self.dataPath, key) + ".csv"

import abc
from abc import abstractmethod

import pandas as pd


class BaseIo(metaclass=abc.ABCMeta):
    # _logger = Logger.getLogger('BaseIo')

    def __init__(self):
        pass

    @abstractmethod
    def saveData(self, key, df: pd.DataFrame):
        """
        存储数据
        :param key: 唯一制定的键，用于存储和读取数据
        :param df: 数据，类型为DataFrame
        :return: None
        """
        pass

    @abstractmethod
    def hasKey(self, key):
        """
        判断已有数据中是否有对应的键
        :param key:
        :return:
        """
        pass

    @abstractmethod
    def readData(self, key) -> pd.DataFrame:
        pass

    @abstractmethod
    def readAllData(self):
        pass

    @abstractmethod
    def getAllKeys(self):
        pass

    @abstractmethod
    def remove(self, key):
        pass

    @abstractmethod
    def clear(self, prefix):
        pass

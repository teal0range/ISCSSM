from collections import defaultdict

from Io import CsvIo
import regex as re
import os
import pandas as pd

io = CsvIo()


def findMergeTarget():
    merge_list = defaultdict(lambda x: 0)
    keys = list(io.getAllKeys())
    keys = sorted(keys)
    for key in keys:
        match = re.findall("([^0-9]+)([0-9]+)$", key)
        if len(match) != 0:
            merge_list.update({match[0][0]: int(match[0][1])})
    return dict(merge_list)


def merge():
    for key, val in findMergeTarget().items():
        base = io.readData(key)
        io.remove(key)
        base = base.append([io.readData(key + str(idx + 1)) for idx in range(val)])
        [io.remove(key + str(idx + 1)) for idx in range(val)]
        io.saveData(key, base)


def convert():
    for root, dirs, files in os.walk("Data/data"):
        for file in files:
            if file.endswith(".xlsx"):
                pd.read_excel("Data/data/{}".format(file), skiprows=[1, 2]). \
                    to_csv("Data/data/{}".format(file[:-4] + "csv"), index=False)


if __name__ == '__main__':
    convert()
    # merge()

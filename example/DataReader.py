"""
A data parser for Porto Seguro's Safe Driver Prediction competition's dataset.
URL: https://www.kaggle.com/c/porto-seguro-safe-driver-prediction
"""
import pandas as pd


class FeatureDictionary(object):
    def __init__(self, trainfile=None, testfile=None,
                 dfTrain=None, dfTest=None, numeric_cols=[], ignore_cols=[]):
        assert not ((trainfile is None) and (dfTrain is None)), "trainfile or dfTrain at least one is set"
        assert not ((trainfile is not None) and (dfTrain is not None)), "only one can be set"
        assert not ((testfile is None) and (dfTest is None)), "testfile or dfTest at least one is set"
        assert not ((testfile is not None) and (dfTest is not None)), "only one can be set"
        self.trainfile = trainfile
        self.testfile = testfile
        self.dfTrain = dfTrain
        self.dfTest = dfTest
        self.numeric_cols = numeric_cols
        self.ignore_cols = ignore_cols
        self.gen_feat_dict()

    def gen_feat_dict(self):
        '''
        产生特征字典。
        把每个特征的取值打平，每个特征对应一个，从0到大的index（段）。多个取值的特征，就对应多个index。

        :return: 一个dict。element可能是整数，也可能是一个dict。前者对于连续特征，后者对于类型特征。
        '''

        # 读取数据。略显多余。
        if self.dfTrain is None:
            dfTrain = pd.read_csv(self.trainfile)
        else:
            dfTrain = self.dfTrain

        if self.dfTest is None:
            dfTest = pd.read_csv(self.testfile)
        else:
            dfTest = self.dfTest

        df = pd.concat([dfTrain, dfTest])
        self.feat_dict = {}
        tc = 0
        for col in df.columns:
            # 忽略列
            if col in self.ignore_cols:
                continue

            if col in self.numeric_cols:
                # 数值类型的特征为什么要给一个index呢？？？
                # map to a single index
                self.feat_dict[col] = tc
                tc += 1
            else:
                # 类别特征
                # 比如numpy.array([2,3,4]), tc=8
                # 那么{2: 8, 3: 9, 4: 10}
                us = df[col].unique()
                self.feat_dict[col] = dict(zip(us, range(tc, len(us)+tc)))
                tc += len(us)

        # 特征的维度——即打平有多少个不同的列，不同的取值
        self.feat_dim = tc


class DataParser(object):
    def __init__(self, feat_dict):
        self.feat_dict = feat_dict

    def parse(self, infile=None, df=None, has_label=False):
        '''
        df分解成两个——dfi和dfv。同时还有y。
        前者是index，后者是value。两者的shape完全一样

        :param infile:
        :param df:
        :param has_label:
        :return:
        '''

        assert not ((infile is None) and (df is None)), "infile or df at least one is set"
        assert not ((infile is not None) and (df is not None)), "only one can be set"

        # 下面的dfi是df处理后的dataframe
        if infile is None:
            dfi = df.copy()
        else:
            dfi = pd.read_csv(infile)

        # 获取y
        if has_label:
            y = dfi["target"].values.tolist()
            dfi.drop(["id", "target"], axis=1, inplace=True)
        else:
            ids = dfi["id"].values.tolist()
            dfi.drop(["id"], axis=1, inplace=True)

        # dfi for feature index
        # dfv for feature value which can be either binary (1/0) or float (e.g., 10.24)

        # 下面的dfi是df处理后的dataframe
        dfv = dfi.copy()
        for col in dfi.columns:
            # 去掉忽略列
            if col in self.feat_dict.ignore_cols:
                dfi.drop(col, axis=1, inplace=True)
                dfv.drop(col, axis=1, inplace=True)
                continue

            # 连续数据，还是类别特征，处理的方法不一样
            if col in self.feat_dict.numeric_cols:
                # 连续数字特征，成了那一列的列号
                dfi[col] = self.feat_dict.feat_dict[col]
            else:
                # 类别特征
                # self.feat_dict.feat_dict[col]返回关于该列的一个dict，如{2: 8, 3: 9, 4: 10}
                # 那么该列的几个类型取值2、3、4就知道第几个序号了（8、9、10）
                # 也就是说，该列的取值是一个int。
                dfi[col] = dfi[col].map(self.feat_dict.feat_dict[col])
                dfv[col] = 1.

        # list of list of feature indices of each sample in the dataset
        Xi = dfi.values.tolist()
        # list of list of feature values of each sample in the dataset
        Xv = dfv.values.tolist()
        if has_label:
            return Xi, Xv, y
        else:
            return Xi, Xv, ids


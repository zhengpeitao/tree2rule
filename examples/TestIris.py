# -*- coding: utf-8 -*-
"""
Created on Thu Apr 29 16:36:05 2021

@author: victorzheng
"""
import sys
sys.path.append("../tree2rule")

import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree
from TreeToRule import *


iris = load_iris()
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)

feature_names=['花萼长','花萼宽','花瓣长','花瓣宽']
# class_names=['山鸢尾','花斑的']
class_names=iris.target_names

# 生成规则
treeUtil = TreeUtil(
    clf=clf,
    feature_names=feature_names,
    class_names=class_names
)
returndata = treeUtil.getRuleList(top_num=10, print_detail=True)
print("done")

 
# 生成dot_data文件
dot_data = tree.export_graphviz(
        clf, 
        feature_names=feature_names,
        class_names=class_names,
        out_file=None
    )
# print(dot_data)


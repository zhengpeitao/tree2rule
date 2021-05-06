# -*- coding: utf-8 -*-
"""
Created on Wed Apr 28 17:58:33 2021

@author: victorzheng
"""
import numpy as np
from sklearn import tree


# 树的叶子节点
TREE_LEAF=-1

class RuleTree():
    """ 规则树 """
    
    def __init__(self, parentNode=None):
        self.parentNode = parentNode
        self.nodes = {}
        self.ranks = {'leaves': []}
        
    
    def getNode(self, node_id):
        return self.nodes.get(str(node_id))
    
    def addNode(self, node):
        self.nodes[str(node.node_id)]=node
        

        
class RuleNode():
    """ 规则树节点 """
    
    def __init__(self, node_id=None, parent_node_id=None, 
                 children_left_node_id=None, children_right_node_id=None):
        self.parent_node_id = parent_node_id
        self.children_left_node_id = children_left_node_id
        self.children_right_node_id = children_right_node_id
        
        self.node_id = node_id
        self.pro_conditon_bool = None
        
        
    def setParent(self, node_id):
        self.parent_node_id = node_id
        
    def setChildrenLeft(self, node_id):
        self.children_left_node_id = node_id
        
    def setChildrenRight(self, node_id):
        self.children_right_node_id = node_id
        

class TreeUtil():
    """ 树提取规则工具 """
    
    def __init__(self, clf, feature_names=None, class_names=None):
        self.clf = clf        
        self.ranks = {'leaves': []}
        self.ruleTree = RuleTree()
        self.feature_names = feature_names
        self.class_names = class_names
        
        self.buildRuleTree(self.clf)


    def getOrCreateTreeNode(self, ruleTree, node_id):
        """ 获取树节点 """
        ruleNode = ruleTree.getNode(node_id)
        if ruleNode == None:    
            ruleNode = RuleNode(node_id=node_id)
            ruleTree.addNode(ruleNode)
            if node_id == 0:
                ruleTree.parentNode = ruleNode
        
        return ruleNode

    def buildRuleTree(self, clf):
        """ 构建树结构 """
        self.recurse(clf.tree_, self.ruleTree, node_id=0, parent=-1)
    

    def recurse(self, tree, ruleTree, node_id, parent, depth=0):
        """ 递归遍历 """
        
        if node_id == TREE_LEAF:
            return
        
        # 当前节点
        currentNode = self.getOrCreateTreeNode(ruleTree, node_id)
        
        # ------------------------------------------
        # 1. 构建节点关系
        # ------------------------------------------   
        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child != TREE_LEAF:
            leftChildNode = self.getOrCreateTreeNode(ruleTree, left_child)
            rightChildNode = self.getOrCreateTreeNode(ruleTree, right_child)
        
            # 构建子节点关系
            currentNode.setChildrenLeft(left_child)
            currentNode.setChildrenRight(right_child)
            # 构建父节点关系
            leftChildNode.setParent(node_id)
            leftChildNode.pro_conditon_bool = True
            rightChildNode.setParent(node_id)
            rightChildNode.pro_conditon_bool = False
        
        # ------------------------------------------
        # 2. 构建节点属性
        # ------------------------------------------   
        # 关系运算符
        relational_operator = '<=' if currentNode.pro_conditon_bool==True else '>'
        # 填充属性
        self.fillNodePro(tree, node_id, relational_operator)    
        

        # ------------------------------------------
        # 递归遍历构建
        # ------------------------------------------   
        if tree.max_depth is None or depth <= tree.max_depth:
    
            # 存储树层级
            if left_child == TREE_LEAF:
                ruleTree.ranks['leaves'].append(str(node_id))
            elif str(depth) not in ruleTree.ranks:
                ruleTree.ranks[str(depth)] = [str(node_id)]
            else:
                ruleTree.ranks[str(depth)].append(str(node_id))
    
            # 递归构建
            if left_child != TREE_LEAF:
                self.recurse(tree, ruleTree, left_child, parent=node_id, depth=depth + 1)
                self.recurse(tree, ruleTree, right_child, parent=node_id, depth=depth + 1)
    
        else:
            ruleTree.ranks['leaves'].append(str(node_id))



    def fillNodePro(self, tree, node_id, relational_operator=None):
        """ 填充属性 """
        
        if tree.n_outputs == 1:
            value = tree.value[node_id][0, :]
        else:
            value = tree.value[node_id]
    
        
        feature = None # 特征名称
        condition = None
        threshold = None # 阈值
        if tree.children_left[node_id] != TREE_LEAF:
            if self.feature_names is not None:
                feature = self.feature_names[tree.feature[node_id]]
    
            threshold = round(tree.threshold[node_id], 2)
            condition = '%s %s %s' % (feature, relational_operator, threshold)
    
    
        # 数量
        node_num = tree.n_node_samples[node_id]
        node_num_pct = tree.n_node_samples[node_id] / float(tree.n_node_samples[0])
    
        # 分类
        class_name = self.class_names[np.argmax(value)]
        
        # 属性
        myNode = self.getOrCreateTreeNode(self.ruleTree, node_id)
        myNode.pro_value = value
        myNode.pro_feature = feature
        myNode.pro_threshold = threshold
        myNode.pro_relational_operator = relational_operator
        myNode.pro_class = class_name
        myNode.pro_condition = condition
        myNode.pro_num = node_num
        myNode.pro_num_pct = node_num_pct
    

    def getLeafToHeadPath(self, ruleTree, node_id, array):
        """ 获取从叶子节点到主节点路径 """
        ruleNode = self.getOrCreateTreeNode(ruleTree, node_id)
        if ruleNode != None:        
            array.append(str(node_id))
            if ruleNode.parent_node_id != None:
                self.getLeafToHeadPath(ruleTree, ruleNode.parent_node_id, array)
     



    def getRuleList(self, top_num=None, filter_num=None, 
                    filter_num_pct=None, filter_class=None, print_detail=None):
        """
        获取规则列表

        Parameters
        ----------
        top_num : TYPE, optional
            返回top n. The default is None.
        filter_num : TYPE, optional
            属性num的最低数量要求. The default is None.
        filter_num_pct : TYPE, optional
            属性num_pct的最低数量要求. The default is None.
        filter_class : TYPE, optional
            属性class类型过滤. The default is None.

        Returns
        -------
        None.

        """
        ruleMap = {}
        matchNodeInfoMap = {}
        
        # 排序
        leafNodes = []
        for node_id in self.ruleTree.ranks['leaves']:  
            leafNode = self.ruleTree.getNode(node_id)
            leafNodes.append(leafNode)
            
        leafNodes.sort(key=lambda x:x.pro_num, reverse=True)
        
        
        
        matchNodes = []
        n = 0
        for leafNode in leafNodes:
            
            n += 1
            
            # 过滤条件
            if filter_num is not None and leafNode.pro_num<filter_num:
                continue
            if filter_num_pct is not None and leafNode.filter_num_pct<filter_num_pct:
                continue
            if filter_class is not None and leafNode.pro_class!=filter_class:
                continue
            if top_num is not None and n > top_num:
                continue

            matchNodes.append(leafNode.node_id)
            node_paths=[]
            self.getLeafToHeadPath(self.ruleTree, leafNode.node_id, node_paths)
                        
            lastNode = None
            ruleStr = ""
            for sub_node_id in reversed(node_paths):
                currentNode = self.ruleTree.getNode(sub_node_id)
                
                if lastNode!=None:
                    condition = '%s %s %s' % (
                        lastNode.pro_feature, 
                        currentNode.pro_relational_operator, 
                        lastNode.pro_threshold
                    )
                    if ruleStr != "":
                        ruleStr+="and"
                    ruleStr+=(" %s "%condition)
                lastNode = currentNode
            
            
            ruleMap[str(leafNode.node_id)]=ruleStr
            
            nodeInfo = {}
            nodeInfo['node_id'] = leafNode.node_id
            nodeInfo['num'] = leafNode.pro_num
            nodeInfo['num_pct'] = round(leafNode.pro_num_pct,4)
            nodeInfo['class'] = leafNode.pro_class
            nodeInfo['rule'] = ruleStr
            matchNodeInfoMap[str(leafNode.node_id)] = nodeInfo
            
        
        return_data = {}
        return_data['match_node_leaves'] = matchNodes
        return_data['match_node_info_map'] = matchNodeInfoMap
        return_data['match_rule_map'] = ruleMap
        
        return_data['all_node_leaves'] = self.ruleTree.ranks['leaves']
        return_data['all_node'] = self.ruleTree
        
        
        if print_detail is True:
            self.printDetail(return_data)
    
        return return_data
    
    
    def getColumnMaxLength(self, return_data, column_name):
        """ 获取列的最大长度 """
        
        match_node_info_map = return_data.get('match_node_info_map')
        match_node_leaves = return_data.get('match_node_leaves')
        
        max_len = len(column_name)
        for match_node_id in match_node_leaves:
            node = match_node_info_map.get(str(match_node_id))
            column_val = node.get(column_name)
            
            column_len = len(str(column_val))
            if column_len > max_len:
                max_len = column_len
                
        return max_len
            
        
    
    def printDetail(self, return_data):
        """ 打印明细 """
        
        match_node_info_map = return_data.get('match_node_info_map')
        match_node_leaves = return_data.get('match_node_leaves')
        
        column_map = {
            'node_id':'', 
            'num':'', 
            'num_pct':'',
            'class':'',
            'rule':''
        }
        
        column_length = {}
        for k,v in column_map.items():
            column_length[k] = self.getColumnMaxLength(return_data, k)
        
        total_length = 0
        for k,v in column_length.items():
            total_length+=v
            
        print("满足条件的节点：%s"%match_node_leaves)
        print("规则明细：")
        
        
        lines = ''
        for k,v in column_map.items():
            s = (v if v!='' else k).ljust(column_length.get(k)+1), 
            lines += '|%s'%s
        print(lines)
        
        for match_node_id in match_node_leaves:
            node = match_node_info_map.get(str(match_node_id))
            
            lines = ''
            for k,v in column_map.items():
                s = str(str(node.get(k)).ljust(column_length.get(k)+1)), 
                lines += '|%s'%s
            
            print(lines)
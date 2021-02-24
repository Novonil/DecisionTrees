import pandas as pd
import math as mt
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import copy
import argparse

start_time = time.time()


def entropy(negatives, positives):
    total = positives + negatives
    if positives == 0 or negatives == 0:
        E = 0
    else:
        E = - (float(float(positives) / float(total))) * mt.log(float((float(positives) / float(total))), 2) - (
            float(float(negatives) / float(total))) * mt.log(float((float(negatives) / float(total))), 2)
    return E


def variance(negatives, positives):
    total = positives + negatives
    if positives == 0 or negatives == 0:
        V = 0
    else:
        V = float(float((negatives * positives)) / float((total * total)))
    return V


def Information_Gain(D_Set, E, check):
    des_attribute = None
    Gain_Max = 0
    zero_list = [0, 0]
    one_list = [0, 0]
    E_pos = 0
    E_neg = 0
    D_arr_T = D_Set.values.T
    for row in D_arr_T[:-1]:
        negatives = [0, 0]
        positives = [0, 0]
        for i in range(len(row)):
            if row[i] == 0 and D_arr_T[-1][i] == 1:
                negatives[1] = negatives[1] + 1
            if row[i] == 0 and D_arr_T[-1][i] == 0:
                negatives[0] = negatives[0] + 1
            if row[i] == 1 and D_arr_T[-1][i] == 1:
                positives[1] = positives[1] + 1
            if row[i] == 1 and D_arr_T[-1][i] == 0:
                positives[0] = positives[0] + 1

        if check == 0:
            E0 = entropy(negatives[0], negatives[1])
            E1 = entropy(positives[0], positives[1])
        else:
            E0 = variance(negatives[0], negatives[1])
            E1 = variance(positives[0], positives[1])

        total = negatives[0] + negatives[1] + positives[0] + positives[1]
        IG = E - ((positives[0] + positives[1]) / total) * E1 - ((negatives[0] + negatives[1]) / total) * E0
        # print(G)

        if Gain_Max < IG:
            Gain_Max = IG
            # print(D_arr_T.tolist().index(col.tolist()))
            des_attribute = D_Set.columns[D_arr_T.tolist().index(row.tolist())]
            E_neg = E0
            E_pos = E1
            zero_list = negatives
            one_list = positives

    return (des_attribute, E_neg, E_pos, zero_list, one_list)


def Predictor(decision_tree, t_set):
    label_list = []
    np_mat = t_set.iloc[:, 0:-1].values
    for row in np_mat.tolist():
        label_list.append(traverse_decision_tree(decision_tree, row))
    label_ser = pd.Series({"Pred_Y": label_list})
    # print(label_ser)
    correct = (t_set.Y == label_ser.Pred_Y).sum()

    acu = correct / len(label_ser.Pred_Y)
    return acu, label_ser


def traverse_decision_tree(decision_tree, row):
    if decision_tree.value == 0:
        return 0
    elif decision_tree.value == 1:
        return 1
    else:
        index = decision_tree.value.replace("X", "")
        val = row[int(index)]
        if val == 0:
            return traverse_decision_tree(decision_tree.left, row)
        else:
            return traverse_decision_tree(decision_tree.right, row)


nodes_at_level = []


def maximum_depth(node):
    if node is None:
        return 0
    else:
        left_depth = maximum_depth(node.left)
        right_depth = maximum_depth(node.right)

        if left_depth < right_depth:
            return right_depth + 1
        else:
            return left_depth + 1


def add_level(decision_tree, level):
    if decision_tree is None:
        return
    if level == 1:
        nodes_at_level.append(decision_tree)
    elif level > 1:
        add_level(decision_tree.left, level - 1)
        add_level(decision_tree.right, level - 1)


def bfs(decision_tree):
    list_of_levels = []
    d = maximum_depth(decision_tree)
    for i in range(1, d + 1):
        add_level(decision_tree, i)
        # temp_l = .copy()
        temp_l = copy.copy(nodes_at_level)
        list_of_levels.append(temp_l)
        nodes_at_level.clear()
    return list_of_levels


def Reduced_Error_Pruning(list_of_levels, v_set, decision_tree, accuracy):
    prev_accuracy = accuracy
    for level in list_of_levels[:-1]:
        for node in level:
            if node.value != 0 and node.value != 1:
                # root node has depth 0
                # new_accuracy = search_node(node, decision_tree, len(list_of_levels) - 1 - list_of_levels.index(level), naive_accuracy)
                # naive_accuracy = new_accuracy
                temp_value = node.value
                temp_left = node.left
                temp_right = node.right
                # print(node.value)
                zeros = node.negatives[0] + node.positives[0]
                ones = node.negatives[1] + node.positives[1]
                if ones > zeros:
                    node.value = 1
                else:
                    node.value = 0
                node.left = None
                node.right = None
                accu_racy, lbl_set = Predictor(decision_tree, v_set)
                if accu_racy < prev_accuracy:
                    node.value = temp_value
                    node.left = temp_left
                    node.right = temp_right
                else:
                    prev_accuracy = accu_racy
    return prev_accuracy


def Depth_Based_Pruning(list_of_levels, s_d_list, v_set, decision_tree):
    acc_list = []
    for d in s_d_list:
        storage_list = []
        if d < len(list_of_levels):
            for n_node in list_of_levels[d]:
                if (n_node.value != 0) and (n_node.value != 1):
                    a = copy.deepcopy(n_node)
                    idx = list_of_levels[d].index(n_node)
                    storage_list.append((a, idx))
                    if (n_node.negatives[1] + n_node.positives[1]) > (n_node.negatives[0] + n_node.positives[0]):
                        n_node.value = 1
                    else:
                        n_node.value = 0
                    n_node.left = None
                    n_node.right = None
            acu, lbl_set = Predictor(decision_tree, v_set)
            acc_list.append(acu)
            # print(len(list_of_levels))
            # print("accuracy on validation set = {}, depth = {}".format(acu, d))

            for a, k in storage_list:
                list_of_levels[d][k].value = a.value
                list_of_levels[d][k].left = a.left
                list_of_levels[d][k].right = a.right

    max_d = s_d_list[acc_list.index(max(acc_list))]
    for n_node in list_of_levels[max_d]:
        if (n_node.value != 0) and (n_node.value != 1):
            if (n_node.negatives[1] + n_node.positives[1]) > (n_node.negatives[0] + n_node.positives[0]):
                n_node.value = 1
            else:
                n_node.value = 0
            n_node.left = None
            n_node.right = None
    return max_d, max(acc_list)


class Node:
    def __init__(self, val, D0=None, D1=None, negatives=None, positives=None):
        self.value = val
        self.left = D0
        self.right = D1
        self.negatives = negatives
        self.positives = positives


def Decision_Tree(D_set, E_parent, check):
    if (len(D_set["Y"].unique()) == 1) and (0 in D_set["Y"].unique()):
        return Node(0)
    elif (len(D_set["Y"].unique()) == 1) and (1 in D_set["Y"].unique()):
        return Node(1)
    # for the last feature remaining just split on it and give its leaf nodes 0 or 1 value
    elif len(D_set.columns) <= 2:
        temporary_zeros = D_set["Y"][D_set.iloc[:, 0] == 0]
        ones_of_zeros = temporary_zeros.sum()
        zeros_of_zeros = len(temporary_zeros) - ones_of_zeros
        negative_list = [zeros_of_zeros, ones_of_zeros]

        temporary_ones = D_set["Y"][D_set.iloc[:, 0] == 1]
        one_ones = temporary_ones.sum()
        one_zeros = len(temporary_ones) - one_ones
        positive_list = [one_zeros, one_ones]
        # if equal no. of 1 and 0 exist in the target column then the value will be 0
        if ones_of_zeros < zeros_of_zeros:
            left = 0
        else:
            left = 1
        if one_ones < one_zeros:
            right = 0
        else:
            right = 1
        return Node(D_set.columns[0], Node(left), Node(right), negative_list, positive_list)
    else:
        des_attribute, E0, E1, negatives, positives = Information_Gain(D_set, E_parent, check)
        # print(des_attribute)
        D0 = D_set[D_set[des_attribute] == 0]
        D1 = D_set[D_set[des_attribute] == 1]
        if len(D0.columns) > 2:
            D0 = D0.drop([des_attribute], axis=1)
        if len(D1.columns) > 2:
            D1 = D1.drop([des_attribute], axis=1)

        return Node(des_attribute, Decision_Tree(D0, E0, check), Decision_Tree(D1, E1, check), negatives, positives)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-algorithm_number', '--algo_no', type=int)
    parser.add_argument('-train_data', '--train_set', type=str)
    parser.add_argument('-valid_data', '--valid_set', type=str)
    parser.add_argument('-test_data', '--test_set', type=str)

    arg = parser.parse_args()
    algo_no = arg.algo_no
    training_set = arg.train_set
    valid_name = arg.valid_set
    testdata_name = arg.test_set

    dataset = pd.read_csv(training_set, header=None)
    validation_set = pd.read_csv(valid_name, header=None)
    test_set = pd.read_csv(testdata_name, header=None)

    for elem in [dataset, validation_set, test_set]:
        dcols = len(elem.columns)
        # drows = len(elem.index)
        elem.columns = ["X{}".format(i) for i in range(dcols)]
        l = list(elem.columns)
        l[-1] = "Y"
        elem.columns = l

    Y = dataset.iloc[:, len(dataset.columns) - 1]
    no_of_pos_Y = int(Y.sum())
    no_of_neg_Y = int(len(dataset.index) - no_of_pos_Y)
    entropy_Y = entropy(no_of_pos_Y, no_of_neg_Y)
    variance_Y = variance(no_of_pos_Y, no_of_neg_Y)

    if algo_no == 1:
        print("Naive decision tree with Entropy Heuristic...")
        check = 0
        decision_tree = Decision_Tree(dataset, entropy_Y, check)
        print("Tree is Ready.")
        accuracy, labels = Predictor(decision_tree, test_set)
        print("The accuracy on test_set = {}".format(accuracy))

    elif algo_no == 2:
        print("Naive Decision Tree with Variance Heuristic.")
        check = 1
        decision_tree = Decision_Tree(dataset, variance_Y, check)
        print("Tree is Ready.")
        accuracy, labels = Predictor(decision_tree, test_set)
        print("The accuracy on test_set = {}".format(accuracy))

    elif algo_no == 3:
        print("Decision Tree with Entropy Heuristic and Reduced Error Pruning.")
        check = 0
        decision_tree = Decision_Tree(dataset, entropy_Y, check)
        print("Tree is Ready.")
        accuracy, labels = Predictor(decision_tree, test_set)
        print("Naive accuracy on test_set = {}".format(accuracy))
        list_of_levels = bfs(decision_tree)
        list_of_levels.reverse()
        re_accuracy = Reduced_Error_Pruning(list_of_levels, validation_set, decision_tree, accuracy)
        post_pruning_accuracy, labels = Predictor(decision_tree, test_set)
        print("Accuracy on Validation Set with Reduced Error Pruning = {}".format(re_accuracy))
        print("Change in accuracy w.r.t Naive = {}".format(re_accuracy - accuracy))
        print("Accuracy on test_set after Reduced Error Pruning = {}".format(post_pruning_accuracy))


    elif algo_no == 4:
        print("Decision Tree with Variance Heuristic and Reduced Error Pruning.")
        check = 1
        decision_tree = Decision_Tree(dataset, variance_Y, check)
        print("Tree is Ready.")
        accuracy, labels = Predictor(decision_tree, test_set)
        print("Naive accuracy on test_set = {}".format(accuracy))
        list_of_levels = bfs(decision_tree)
        list_of_levels.reverse()
        re_accuracy = Reduced_Error_Pruning(list_of_levels, validation_set, decision_tree, accuracy)
        post_pruning_accuracy, labels = Predictor(decision_tree, test_set)
        print("Accuracy on Validation Set with Reduced Error Pruning = {}".format(re_accuracy))
        print("Change in accuracy w.r.t Naive = {}".format(re_accuracy - accuracy))
        print("Accuracy on test_set after Reduced Error Pruning = {}".format(post_pruning_accuracy))


    elif algo_no == 5:
        print("Decision Tree with Entropy Heuristic and Depth Based Pruning.")
        check = 0
        decision_tree = Decision_Tree(dataset, entropy_Y, check)
        print("Tree is Ready.")
        accuracy, labels = Predictor(decision_tree, test_set)
        print("Naive accuracy on test_set = {}".format(accuracy))
        stop_depths = [5, 10, 15, 20, 50, 100]
        list_of_levels = bfs(decision_tree)
        max_d, depth_accuracy = Depth_Based_Pruning(list_of_levels, stop_depths, validation_set, decision_tree)
        depth_prune_post, db_lab = Predictor(decision_tree, test_set)
        print("Depth_max = {}, accuracy = {}".format(max_d, depth_accuracy))
        print("Post pruning accuracy on test set: {} ".format(depth_prune_post))


    elif algo_no == 6:
        print("Decision Tree with Variance Heuristic and Depth Based Pruning.")
        check = 1
        decision_tree = Decision_Tree(dataset, variance_Y, check)
        print("Tree is Ready.")
        accuracy, labels = Predictor(decision_tree, test_set)
        print("Naive accuracy on test_set = {}".format(accuracy))
        stop_depths = [5, 10, 15, 20, 50, 100]
        list_of_levels = bfs(decision_tree)
        max_d, depth_accuracy = Depth_Based_Pruning(list_of_levels, stop_depths, validation_set, decision_tree)
        depth_prune_post, db_lab = Predictor(decision_tree, test_set)
        print("Depth_max = {}, accuracy = {}".format(max_d, depth_accuracy))
        print("Post pruning accuracy on test set: {} ".format(depth_prune_post))


    elif algo_no == 7:
        random_forest = RandomForestClassifier(n_estimators=100, random_state=1)
        random_forest.fit(dataset.iloc[:, 0:-1].values, dataset["Y"].values)
        # predictions
        random_forest_predict = random_forest.predict(test_set.iloc[:, 0:-1].values)
        print("Random Forest Accuracy:", metrics.accuracy_score(test_set["Y"].values, random_forest_predict))


main()

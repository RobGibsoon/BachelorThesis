# from multiprocessing import Pool, cpu_count
#
# import numpy as np
# from cyged.graph_pkg_core import GED
# from cyged.graph_pkg_core import Graph
# from cyged.graph_pkg_core.edit_cost.edit_cost_vector import EditCostVector
# from cyged.graph_pkg_core.graph.edge import Edge
# from cyged.graph_pkg_core.graph.label.label_edge import LabelEdge
# from cyged.graph_pkg_core.graph.label.label_node_vector import LabelNodeVector
# from cyged.graph_pkg_core.graph.node import Node
# from sklearn import svm
# from sklearn.metrics import accuracy_score todo: rollback
# from sklearn.model_selection import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.svm import SVC
# from torch_geometric.data import Data
# from torch_geometric.datasets import TUDataset
#
# from utils import append_hyperparams_file, save_preds, append_accuracies_file, log, get_csv_idx_split
#
# DIR = "references"
#
#
# class ReferenceClassifier:
#     def __init__(self, dataset_name):
#         self.dataset_name = dataset_name
#         self.data = TUDataset(root=f'/tmp/{self.dataset_name}', name=f'{self.dataset_name}')
#         filter_split = get_csv_idx_split(self.dataset_name, "filter")
#         self.X = [self.data[idx] for idx in filter_split]
#         self.y = np.array([self.X[i].y.item() for i in range(len(self.X))])
#         train_split = get_csv_idx_split(self.dataset_name, "train")
#         test_split = get_csv_idx_split(self.dataset_name, "test")
#         train_graphs = [self.X[idx] for idx in train_split]
#         test_graphs = [self.X[idx] for idx in test_split]
#         self.y_train = [self.y[idx] for idx in train_split]
#         self.y_test = [self.y[idx] for idx in test_split]
#         alpha_values = np.arange(0.05, 1.0, 0.1)
#         self.kernelized_data_training = [create_custom_metric(train_graphs, train_graphs, alpha) for alpha in
#                                          alpha_values]
#         log(f'Finished generating train-data kernel for {self.dataset_name}', DIR)
#         self.kernelized_data_test = [create_custom_metric(test_graphs, train_graphs, alpha) for alpha in alpha_values]
#         log(f'Finished generating test-data kernel for {self.dataset_name}', DIR)
#
#     def predict_knn(self):
#         """train and predict with knn"""
#         log(f'running references on: ({self.dataset_name}, knn)', DIR)
#         best_kernel_index = 0
#         prev_score = 0
#         best_knn = None
#         best_grid_search = None
#
#         k_range = list(range(1, 31))
#         param_grid = {'algorithm': ['brute'],
#                       'n_neighbors': k_range}
#         log('Finding best alpha on ptc_mr', DIR)
#         for i, cur_kernel in enumerate(self.kernelized_data_training):
#             clf_knn = KNeighborsClassifier(metric='precomputed')
#
#             # perform hyper parameter selection
#             grid_search = GridSearchCV(clf_knn, param_grid, cv=10, scoring='accuracy', return_train_score=False,
#                                        verbose=1, n_jobs=-1)
#             log(f'Completing knn gridsearch on {self.dataset_name}: ({i + 1}/{len(self.kernelized_data_training)}) ',
#                 DIR)
#             grid_search.fit(cur_kernel, np.ravel(self.y_train))
#             log(f'Completed knn gridsearch on {self.dataset_name}: ({i + 1}/{len(self.kernelized_data_training)}) ',
#                 DIR)
#
#             # construct, train optimal model and perform predictions
#             clf_knn = KNeighborsClassifier(algorithm=grid_search.best_params_['algorithm'],
#                                            n_neighbors=grid_search.best_params_['n_neighbors'],
#                                            metric='precomputed')
#
#             clf_knn.fit(cur_kernel, np.ravel(self.y_train))
#             score = clf_knn.score(cur_kernel, np.ravel(self.y_train))
#             if score > prev_score:
#                 prev_score = score
#                 best_knn = clf_knn
#                 best_kernel_index = i
#                 best_grid_search = grid_search
#         log(f'finished knn fitting on {self.dataset_name} and found best alpha {np.arange(0.05, 1.0, 0.1)[best_kernel_index]}',
#             DIR)
#         append_hyperparams_file(False, best_grid_search, best_knn, self.dataset_name, DIR, ref=True)
#         predictions = best_knn.predict(self.kernelized_data_test[best_kernel_index])
#         test_accuracy = accuracy_score(self.y_test, predictions) * 100
#         save_preds(predictions, self.y_test, type(best_knn).__name__, self.dataset_name, False, ref=True)
#         return test_accuracy
#
#     def predict_svm(self):
#         """train and predict with svm"""
#         log(f'running references on: ({self.dataset_name}, knn)', DIR)
#         best_kernel_index = 0
#         prev_score = 0
#         best_alpha = 0.05
#         small_param_grid = {'C': [0.01, 0.1, 1, 10]}
#         self.kernelized_data_training = [data_training * (-1) for data_training in
#                                          self.kernelized_data_training]
#         self.kernelized_data_test = [data_test * (-1) for data_test in
#                                      self.kernelized_data_test]
#
#         # find best alpha with less extensive param_grid
#         for i, cur_kernel in enumerate(self.kernelized_data_training):
#             clf_svm = svm.SVC(kernel='precomputed')
#
#             # perform hyper parameter selection
#             grid_search = GridSearchCV(clf_svm, small_param_grid, cv=5, scoring='accuracy', error_score='raise',
#                                        return_train_score=False, verbose=1, n_jobs=-1)
#             log(f'Completing svm girdsearch with small param grid on {self.dataset_name}({i + 1}/{len(self.kernelized_data_training)})',
#                 DIR)
#             grid_search.fit(cur_kernel, np.ravel(self.y_train))
#
#             # construct, train optimal model and perform predictions
#             clf_svm = SVC(C=grid_search.best_params_['C'],
#                           kernel='precomputed')
#
#             clf_svm.fit(cur_kernel, np.ravel(self.y_train))
#             score = clf_svm.score(cur_kernel, np.ravel(self.y_train))
#             if score > prev_score:
#                 prev_score = score
#                 best_kernel_index = i
#         log(f'Completed small svm girdsearch on {self.dataset_name} with small param grid and found {np.arange(0.05, 1.0, 0.1)[best_kernel_index]}',
#             DIR)
#
#         # do more detailed optimization now that we have the best kernel (alpha)
#         big_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
#         clf_svm = svm.SVC(kernel='precomputed')
#         detailed_grid_search = GridSearchCV(clf_svm, big_param_grid, cv=10, scoring='accuracy', error_score='raise',
#                                             return_train_score=False, verbose=1, n_jobs=-1)
#         log(f'Completing svm girdsearch on {self.dataset_name} with detailed param grid.', DIR)
#
#         detailed_grid_search.fit(self.kernelized_data_training[best_kernel_index], np.ravel(self.y_train))
#         log(f'finished detailed svm gridsearch on {self.dataset_name}', DIR)
#         append_hyperparams_file(False, f"best alpha: {best_alpha}", clf_svm, self.dataset_name, DIR, ref=True)
#         append_hyperparams_file(False, detailed_grid_search, clf_svm, self.dataset_name, DIR, ref=True)
#
#         # do final fitting with best params
#         best_svm = SVC(C=detailed_grid_search.best_params_['C'],
#                        kernel='precomputed')
#         log(f'fiting best_svm on {self.dataset_name}', DIR)
#         best_svm.fit(self.kernelized_data_training[best_kernel_index], np.ravel(self.y_train))
#         predictions = best_svm.predict(self.kernelized_data_test[best_kernel_index])
#         test_accuracy = accuracy_score(self.y_test, predictions) * 100
#         save_preds(predictions, self.y_test, type(best_svm).__name__, self.dataset_name, False, ref=True)
#
#         return test_accuracy
#
#     def predict_ann(self, device):
#         return 100.0, 3.0, 2.0
#
#
# def create_custom_metric_parallel(test, train, alpha):
#     rows = len(test)
#     cols = len(train)
#     res_mat = np.zeros((rows, cols))
#
#     # Create a pool of worker processes
#     with Pool(cpu_count() - 3) as pool:
#         # Construct the arguments for each task
#         tasks = [(i, test[i], train, alpha) for i in range(rows)]
#
#         # Use imap_unordered to get the results as they become available
#         for i, res_row in pool.imap_unordered(compute_row, tasks):
#             # Store the result in the appropriate row of the matrix
#             res_mat[i] = res_row
#
#     res_mat = res_mat + res_mat.T - np.diag(np.diag(res_mat))
#     return res_mat
#
#
# def create_custom_metric(test, train, alpha):
#     rows = len(test)
#     cols = len(train)
#     res_mat = np.zeros((rows, cols))
#     for i in range(rows):
#         for j in range(cols):
#             res_mat[i, j] = graph_edit_distance(test[i], train[j], alpha)
#     assert np.abs(np.sum(res_mat)) > 0
#     return res_mat
#
#
# def compute_row(args):
#     i, test_i, train, alpha = args
#     cols = len(train)
#     res_row = np.zeros(cols)
#     for j in range(cols):
#         res_row[j] = graph_edit_distance(test_i, train[j], alpha)
#     return i, res_row
#
#
# def graph_edit_distance(gr_1, gr_2, alpha):
#     ged = GED(EditCostVector(1., 1., 1., 1., "euclidean", alpha=alpha))
#     ant_gr_1 = data_to_custom_graph(gr_1)
#     ant_gr_2 = data_to_custom_graph(gr_2)
#     edit_cost = ged.compute_edit_distance(ant_gr_1, ant_gr_2)
#     return edit_cost
#
#
# def data_to_custom_graph(data: Data):
#     n = data.num_nodes
#     m = data.num_edges
#
#     graph = Graph("", "", n)
#
#     # Add nodes to the custom graph
#     for i, node_feat in enumerate(data.x.numpy().astype(np.double)):
#         graph.add_node(Node(i, LabelNodeVector(node_feat)))
#
#     # Add edges to the custom graph
#     edge_index = data.edge_index.numpy()
#     for i in range(0, edge_index.shape[1], 2):
#         src, dest = edge_index[:, i]
#         graph.add_edge(Edge(src, dest, LabelEdge(0)))
#
#     return graph
#
#
# if __name__ == "__main__":
#     # use this for developing
#     dataset_name = "PTC_MR"
#
#     reference_classifier = ReferenceClassifier(dataset_name)
#     svm_acc = reference_classifier.predict_svm()
#     knn_acc = reference_classifier.predict_knn()
#     print(svm_acc)
#     print(knn_acc)
#
#     append_accuracies_file(dataset_name, "knn", False, knn_acc, DIR, ref=True)
#     append_accuracies_file(dataset_name, "svm", False, svm_acc, DIR, ref=True)

## How to use dataset.py

Use the main method in the file itself (at the bottom).
```if __name__ == "__main__": ...```
You may take out or leave in any of the indices in the variable ```wanted_indices```
You may pick any of the TUDdatasets integrated in pytorch (https://chrsmrrs.github.io/datasets/docs/datasets/). We used
_PTC_MR_, _Mutagenicity_ and _MUTAG_. Note that if the graphs in the dataset are not connected then they will be
filtered,
this means only connected graphs are stored in the created CSVs.
For example replace _Mutagenicity_ with whatever dataset you wish to work on at the following place in the code:
`dataset = TUDataset(root='/tmp/Mutagenicity', name='Mutagenicity')`

## How to use embedding_classifier.py

### Expected arguments:

Just pass an index from 0-28. As an example you can us the code line:
```python pythonProject/embedding_classifier.py --idx=10```
A list of which index will do what:
First entry is the dataset name, second one the classfier to be used, third one if feature selection should be used.
The last entry is to generate reference values, you will most likely not need this.

* 0: ("PTC_MR", "ann", True, False),
* 1: ("PTC_MR", "knn", True, False),
* 2: ("PTC_MR", "svm", True, False),
* 3: ("Mutagenicity", "ann", True, False),
* 4: ("Mutagenicity", "knn", True, False),
* 5: ("Mutagenicity", "svm", True, False),
* 6: ("MUTAG", "ann", True, False),
* 7: ("MUTAG", "knn", True, False),
* 8: ("MUTAG", "svm", True, False),
* 9: ("PTC_MR", "ann", False, False),
* 10: ("PTC_MR", "knn", False, False),
* 11: ("PTC_MR", "svm", False, False),
* 12: ("Mutagenicity", "ann", False, False),
* 13: ("Mutagenicity", "knn", False, False),
* 14: ("Mutagenicity", "svm", False, False),
* 15: ("MUTAG", "ann", False, False),
* 16: ("MUTAG", "knn", False, False),
* 17: ("MUTAG", "svm", False, False),
* 18: ("PTC_MR", "ann", None, True),
* 19: ("PTC_MR", "knn", None, True),
* 20: ("PTC_MR", "svm", None, True),
* 21: ("Mutagenicity", "ann", None, True),
* 22: ("Mutagenicity", "knn", None, True),
* 23: ("Mutagenicity", "svm", None, True),
* 24: ("MUTAG", "ann", None, True),
* 25: ("MUTAG", "knn", None, True),
* 26: ("MUTAG", "svm", None, True),
* 27: ("PTC_MR_bc", "svm", False, False),
* 28: ("PTC_MR_bw", "svm", False, False),

### (Ignore following section)

Ignore the following section, it was for a older vesion of embedding_classifier.py, but might be used again in the
future so I
didn't delte it.

* dn: The name of the dataset to classify. Choose an option from **PTC_MR, MUTAG, NCI1, Mutagenicity**
* clf: The classifier used to classify. Choose an option from **ann, knn or svm**. Default is knn.
* fs: Is a boolean flag, if included then classification will be made with feature selection, if left out then no
  feature selection will occur.
* ref: Is a boolean flag, if included then it generates the reference values.

To use embedding_classifier.py you will need to run it from the BachelorThesis directory.
After running this a file the labels and predictions are stored in the predictions folder. It will also add the
resulting accuracy in the accuracies.txt file. Furthermore the hyper parameters and (if --fs is used) the features will
be stored.

### Examples:

* ```python pythonProject/embedding_classifier.py --dn=PTC_MR --clf=svm --fs``` This will classify the PTC_MR dataset
  with feature
  selection using an SVM.
* ```python pythonProject/embedding_classifier.py --dn=PTC_MR --clf=ann``` This will classify the MUTAG dataset without
  feature
  selection using an ANN.
* ```python pythonProject/embedding_classifier.py --dn=PTC_MR --clf=knn --ref``` This will create the reference values
  of the dataset PTC_MR and classifier KNN using the graph edit distance.

---

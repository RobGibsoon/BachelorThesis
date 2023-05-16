## How to use dataset.py
Use the main method in the file itself (at the bottom). 
```if __name__ == "__main__": ...```
You may take out or leave in any of the indices in the variable ```wanted_indices```
You may pick any of the TUDdatasets integrated in pytorch (https://chrsmrrs.github.io/datasets/docs/datasets/). We used
_PTC_MR_, _Mutagenicity_ and _MUTAG_. Note that if the graphs in the dataset are not connected then they will be filtered, 
this means only connected graphs are stored in the created CSVs.
For example replace _Mutagenicity_ with whatever dataset you wish to work on at the following place in the code:
`dataset = TUDataset(root='/tmp/Mutagenicity', name='Mutagenicity')`


## How to use embedding_classifier.py

### Expected arguments:
* dn: The name of the dataset to classify. Choose an option from **PTC_MR, MUTAG, NCI1, Mutagenicity**
* clf: The classifier used to classify. Choose an option from **ann, knn or svm**. Default is knn.
* fs: Is a boolean flag, if included then classification will be made with feature selection, if left out then no 
feature selection will occur.
* ref: Is a boolean flag, if included then it generates the reference values.

To use embedding_classifier.py you will need to run it from the BachelorThesis directory.
After running this a file the labels and predictions are stored in the predictions folder. It will also add the 
resulting accuracy in the accuracies.txt file. Furthermore the hyper parameters and (if --fs is used) the features will be stored.

### Examples:
* ```python pythonProject/embedding_classifier.py --dn=PTC_MR --clf=svm --fs``` This will classify the PTC_MR dataset with feature 
selection using an SVM.
* ```python pythonProject/embedding_classifier.py --dn=PTC_MR --clf=ann``` This will classify the MUTAG dataset without feature 
selection using an ANN.
* ```python pythonProject/embedding_classifier.py --dn=PTC_MR --clf=knn --ref``` This will create the reference values 
of the dataset PTC_MR and classifier KNN using the graph edit distance.
---

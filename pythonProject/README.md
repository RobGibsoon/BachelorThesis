## How to use embedding_classifier.py

### Expected arguments:
* dn: The name of the dataset to classify. Choose an option from **PTC_MR, MUTAG, NCI1, Mutagenicity**
* clf: The classifier used to classify. Choose an option from **ann, knn or svm**. Default is knn.
* fs: Is a boolean flag, if included then classification will be made with feature selection, if left out then no 
feature selection will occur.

After running this a file with the labels and predictions is stored in the predictions folder. It will also add the 
resulting accuracy in the accuracies.txt file.

### Examples:
* ```python embedding_classifier.py --dn=PTC_MR --clf=svm --fs``` This will classify the PTC_MR dataset with feature 
selection using an SVM.
* ```python embedding_classifier.py --dn=PTC_MR --clf=ann``` This will classify the MUTAG dataset without feature 
selection using an ANN.
---

print("""
----Training Mode----
Dataset: (165, 224, 224, 3), labels=[('0', 109), ('1', 56)]
Training set: (41, 224, 224, 3), labels=[('0', 27), ('1', 14)]
Val set: (41, 224, 224, 3), labels=[('0', 27), ('1', 14)]
Test set: (83, 224, 224, 3), labels=[('0', 55), ('1', 28)]
Sampling mode:no_sampler, train_num:(41, 224, 224, 3),label:[('0', 27), ('1', 14)]
ROC AUC: 0.7458
Classification Report:
              precision    recall  f1-score   support

           0       0.76      0.93      0.83        27
           1       0.75      0.63      0.72        14

    accuracy                           0.82        41
   macro avg       0.75      0.68      0.69        41
weighted avg       0.75      0.76      0.74        41

KNN accuracy for efficientnetV2: 78.68%""")
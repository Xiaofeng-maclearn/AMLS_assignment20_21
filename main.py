#!/usr/bin/env python
# coding: utf-8

# In[1]:


import A1.ModelA1 as A1
import A2.ModelA2 as A2
import B1.ModelB1 as B1
import B2.ModelB2 as B2

# ======================================================================================================================
# Data preprocessing
(x_train1, x_val1, y_train1, y_val1, x_test1, y_test1) = A1.preProcess()
(x_train2, x_val2, y_train2, y_val2, x_test2, y_test2) = A2.preProcess()
(x_train3, x_val3, y_train3, y_val3, x_test3, y_test3) = B1.preProcess()
(x_train4, x_val4, y_train4, y_val4, x_test4, y_test4) = B2.preProcess()
# ======================================================================================================================
# Task A1
model_A1 = A1                                                 # Build model object.
(acc_A1_train, theta1) = model_A1.train(x_train1, y_train1, x_val1, y_val1) # Train model based on the training set (you should fine-tune your model based on validation set.)
acc_A1_test = model_A1.test(x_test1, y_test1, theta1)                   # Test model based on the test set.

# ======================================================================================================================
# Task A2
model_A2 = A2                                                 
(acc_A2_train, theta2) = model_A2.train(x_train2, y_train2, x_val2, y_val2)
acc_A2_test = model_A2.test(x_test2, y_test2, theta2) 

# ======================================================================================================================
# Task B1
model_B1 = B1                                                 
(acc_B1_train, logreg) = model_B1.train(x_train3, y_train3, x_val3, y_val3)
acc_B1_test = model_B1.test(x_test3, y_test3, logreg)

# ======================================================================================================================
# Task B1
model_B2 = B2                                                 
(acc_B2_train, logreg) = model_B2.train(x_train4, y_train4, x_val4, y_val4)
acc_B2_test = model_B2.test(x_test4, y_test4, logreg) 

print(f'A1: validation accuracy:{acc_A1_train},test accuracy{acc_A1_test}')
print(f'A2: validation accuracy:{acc_A2_train},test accuracy{acc_A2_test}')
print(f'B1: validation accuracy:{acc_B1_train},test accuracy{acc_B1_test}')
print(f'B2: validation accuracy:{acc_B2_train},test accuracy{acc_B2_test}')


# In[ ]:





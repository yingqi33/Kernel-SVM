# kernel-SVM

In machine learning, kernel methods are a class of algorithms for pattern analysis. 
The kernel trick aims to transform data into another dimension which has a clear dividing margin between classes of data.
In this way, kernel functions enable them to operate in a high-dimensional, implicit feature space without ever computing 
the coordinates of the data in that space, but rather by simply computing the inner products between the images of all pairs of data in the feature space. 
This approach simplifies computational complexity compared with explicit computation of the coordinates, which is also known as "simple" method.

This file implements simple SVM, kernel SVM (gaussian) from scratch for comparison.

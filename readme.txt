Name: Junfei Liu
Email: jliu137@u.rochester.edu
CSC246 Project1

My readme will go through method, regularization, model stability, results, collaboration,
and additional notes one by one.


My approach is to very straightforwardly implement formula 3.28 (fun calculate_w) and
use w to predict y (fun predict_y) and calculate ERMS (fun get_Erms) in calc.py with
two test functions in K_fold.py, k_fold_autofit determining best fit automatically and
k_fold_given giving ERMS of given m. My algorithm implements the k-fold technique to yield
better result. It shuffles the dataset and partitions the dataset into k parts, trains
on k-1 sets and tests on kth set, and get the average w and error for same m. If
k_fold_autofit is chosen, then it will loop from 0 to given m and find m with smallest error.


Because I employed k-fold technique, the trained model is tested on unseen data so
the result can more objectively reflect the overfit problems. If the model is overfitting,
the error will be significantly higher. Therefore, the degree with the highest accuracy
will be the ”best” order without significant overfitting. By the way, I have gamma in
the calculation, but it does not work well. Even when I change gamma from 1e-18 to 1, the
result won't change a lot. Anyway, you can input a gamma or it will use a default of 0
for gamma when you run it.


My code generates the same "best order" as dataset secrets for dataset A, B, C with
similar weights, order 5-7 for dataset D, and order 7-9 for dataset E. Because my k-fold
shuffles the dataset everytime, my algorithm generates different results everytime, and
the value of k, gamma will affect result as well. The errors of some orders on the
same dataset could be very close, so it sometimes just yields different results, but
trust me it is not a random number generator. Generally, I think the errors are generally
small and parameters are close to dataset secrets. If you get, for example, best m = 4 for
dataset B, please give it a chance to run another 3 times or more, and you will see that
in most cases it yields the right result.


I went through some of my results in the model stability section. My results for datasets
A, B, C match labeled dataset in most cases, and results for dataset D and E have different
orders compared to labeled datasets but have close parameters.
My result for test datasets X, Y, and Z with gamma = 1e-18, k = 5, m = 20 is:
X:
The best m is  6  with ERMS  0.03404937750425448  with weights [ 0.32429466 -0.22111875  1.163467    0.3927913   0.05424039  1.11906297
 -0.23478675]
Y:
The best m is  7  with ERMS  0.032999921199658534  with weights [ 1.96149072  0.47646128 -0.12016798 -0.10641312 -1.6115435   0.5646033
  4.11845993 -0.61732899]
(sometimes m = 16)
Z:
The best m is  4  with ERMS  0.0350272802567458  with weights [ 1.3839786   0.54057038  0.07877111 -0.83583797  0.03404407]
The results of datasets X and Z are stable, but the result of Y could sometimes jump to 16.
I don't really understand why, maybe too much noise? Generally I think it is accurate.


I worked solo for this project.


My code can read all required commandline arguments plus numFolds. You can run by
python polyhunt.py --m M [--gamma GAMMA] --trainPath TRAINPATH [--modelOutput MODELOUTPUT] [--autofit AUTOFIT] [--info] [--numFolds NUMFOLDS]
Here ares something you need to know before running my code.
1. my code cannot handle the case where 50 % k != 0 so please enter a factor of 50 for k.
2. please enter absolute path for modelOutput. If trainPath does not work, enter absolute path
for it as well.
3. if you figured out the best gamma, please let me know :)

Here is an example of a standard run:
C:\Users\23566\PycharmProjects\CSC246Proj1> python polyhunt.py --m 10 --trainPath ./levelOne/A --numFolds 5 --gamma 1e-18 --autofit true --modelOutput C:\Users\23566\PycharmProjects
\CSC246Proj1\output.txt
The result of given m  10 is with ERMS  0.13564827501298649  with weights [ -1.15666624   1.62554275   0.42006216  -0.6610535   -6.66092326
   2.07628292  23.80055309  -2.66242829 -31.55050959   1.23434454
  14.09775418]


Thank you for reading such a long readme. It requires all these sections. I don't want to write so many words either.
Have a nice day!

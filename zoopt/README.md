## 使用方法
solution_list = ExpOpt.min(objective, parameter, lower_dim=0, upper_dim=1) 
需要传入的是lower_dim和upper_dim， 例如希望第一个维度的值始终小于第二个维度，lower_dim=0， upper_dim=1 ，希望第一个维度大于第二个维度，那就传入upper_dim=0， lower_dim=1

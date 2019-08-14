'''
给定数组A[0,1,...,n-1] 构建数组B[0,1,2...,n-1] 其中B[i]=A[0]*....A[i-1]*A[i+1]*...*A[n-1]
'''
class Solution:
    def multiply(self, A):
        C =[1]
        for i in range(len(A)-1):
            C.append(C[-1] * A[i])
        D =[1]
        for j in range(len(A)-1, 0, -1):
            D.append(D[-1] * A[j])
        D.reverse()
        return [C[i] * D[i] for i in range(len(A))]
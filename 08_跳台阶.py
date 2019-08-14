'''
每次1级或者2级
求跳上n级总共有集中跳法

f(1)=1,f(2)=2,f(n)=f(n-1)+f(n-2)
因为n级可以先减去1就是f(n-1),还可以先减去2
'''
class Solution:
    def jumpFloor(self,n):
        a, b = 1, 2
        res = 0
        if n <= 0:
            res = 0
        elif n == 1:
            res = a
        elif n == 2:
            res = b
        else:
            for i in range(n-2):
                res = a+b
                a = b
                b = res
                #这里跟fib一样，完全可以写成 a, b = b, a+b
        return res
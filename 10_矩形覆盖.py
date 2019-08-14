'''
2*1的矩形，使用n个2*1的矩形无重叠覆盖2*n的矩形
总共有几种方法？
可以先除去一个竖着的2*1，f(n-1),或者除去2个横着的1*2，f(n-2)
f(n) = f(n-1)+f(n-2)
'''
class Solution:
    def rectCover(self,number):
        a ,b = 1, 1
        if number <3:
            ans = number
        else:
            for i in range(number-1):
                ans = a+b
                a=b
                b=ans
        return  ans
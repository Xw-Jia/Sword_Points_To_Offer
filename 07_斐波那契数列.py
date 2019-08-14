'''
从0开始，第0项为0
输入整数n，输出第n项
'''
class Solution:
    def Fib(self,n):
        a = 0
        b = 1
        for _ in range(n):
            a, b = b, a+b
        return a
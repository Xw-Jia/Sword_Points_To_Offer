'''
可以跳1，2，3.。。n阶
总共有几种不同跳法？
这种情况下加一级，就会增加一倍
思考下f(n) = 2*f(n-1)
'''
class Solution:
    def jumpFloor2(self,number):
        ans = 1
        if number >= 2:
            for i in range(number-1):
                ans = ans * 2
        return ans
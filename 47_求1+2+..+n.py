'''
要求不能用乘除法，for while if else switch case等判断和条件语句
'''
#写个递归，不过终止条件也类似于判断语句了
class Solution:
    def Sum_solution(self, n):
        return n and (n + self.Sum_solution(n-1))
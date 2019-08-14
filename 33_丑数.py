'''
把只包含质因子2/3/5的数叫做丑数 把1作为第一个丑数
求从小到大顺序的第N个丑数
'''
class Solution:
    def GetUglyNumbers_Solution(self, n):
        if n == 0:
            return 0
        q = [1]
        t2 = t3 = t5 = 0
        for _ in range(n-1):
            a2, a3, a5 = q[t2]*2, q[t3]*3, q[t5]*5
            to_add = min(a2, a3, a5)
            q.append(to_add)
            if a2 == to_add:
                t2 += 1
            if a3 == to_add:
                t3 += 1
            if a5 == to_add:
                t5 += 1
        return q[-1]
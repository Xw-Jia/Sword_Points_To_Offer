'''
求出1-n中1出现的次数
'''
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        countr, i = 0, 1
        while i < n:
            divider = i*10
            countr += (n // divider) * i + min(max(n % divider -i + 1, 0), i)
            i *= 10
        return countr
s = Solution()
res = s.NumberOf1Between1AndN_Solution(156)
#完全看不懂，去一边去吧
class Solution:
    def NumberOf1Between1AndN_Solution(self, n):
        count = 0
        for i in range(1, n+1):
            while i:
                if i%10==1:
                    count += 1
                i /= 10
        return count

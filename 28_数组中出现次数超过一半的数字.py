'''
数组张有一个数字出现的次数超过数组长度的一半，找出这个数字，不存在就输出0
'''
#把这个 叫做波亦尔摩尔投票算法
#这个答案是错的
class Solution:
    def MoreThanHalfNum_Solution(self, numbers):
        count = 0
        candidate = None
        for num in numbers:
            if count == 0:
                candidate = num
            count += (1 if num == candidate else -1)
        return candidate
s = Solution()
c = s.MoreThanHalfNum_Solution(['a','a','b','c'])
'''
使用hash，key是数字，value是次数
'''
class Solution:
    def MoreThanHalfNuma_Solution(self, nums):
        hashs = dict()
        length = len(nums)
        for n in nums:
            hashs[n] = hashs[n]+ 1 if hashs.get(n) else 1
            if hashs[n] > length/2:
                return n
        return 0
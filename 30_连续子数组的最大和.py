'''
给定数组有正负，求出连续子数组的最大和
'''
class Solution:
    def FindGreatestSumOfSubArray(self, nums):
        cp_nums = nums[:]
        for i in range(1, len(nums)): #从index==1开始，判断前一个值是否大于0，大则累加
            if cp_nums[i-1] > 0:
                cp_nums[i] += cp_nums[i-1]
        return max(cp_nums)
lst = [1,-2,3,-2]
s = Solution()
res = s.FindGreatestSumOfSubArray(lst)
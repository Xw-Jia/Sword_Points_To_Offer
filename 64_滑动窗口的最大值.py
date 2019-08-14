'''
输入数列和窗口大小，输出滑动窗口的最大值
'''
class Solution:
    def maxInWindows(self, nums, size):
        return [max(nums[i: i+size])
                for i in range(len(nums)-size+1) if size != 0]
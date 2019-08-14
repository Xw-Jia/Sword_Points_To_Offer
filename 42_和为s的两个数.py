'''
输入一个递增排序的数组和一个数字s，找出数组中的两个数，使和为s，如果有多对，输出积最小的
双指针起点终点移动
'''
class Solution:
    def FindNumberWithSum(self, array, tsum):
        l, r = 0, len(array)-1
        while l < r:
            if array[l]+array[r] < tsum:
                l += 1
            elif array[l]+array[r] > tsum:
                r -= 1
            else:
                return array[l], array[r]
        return []
'''
统计一个数字在排序数组中出现的次数
二分查找？
'''
class Solution:
    def GetNumberOfK(self, data, k):
        def search(n):
            lo, hi = 0, len(data)
            while lo < hi:
                mid = (lo + hi) // 2
                if data[mid] >= n:
                    hi = mid
                else:
                    lo = mid + 1
            return lo
        lo = search(k)
        if k in data[lo : lo+1]:
            return search(k+1)-lo
        else:
            return 0

'''
输入n个整数，找出其中最小的k个数
基于划分，使比第k个数小的都在左边，大的都在右边,递归构建快排
'''
class Solution:
    def GetLeastNumbers_Solution(self, tinput, k):
        if not tinput or k>len(tinput):
            return []
        tinput = self.quick_sort(tinput)
        return tinput[:k]
    def quick_sort(self, lst):
        if not lst:
            return []
        pivot = lst[0]
        left = self.quick_sort([x for x in lst[1:] if x<pivot])
        right = self.quick_sort([x for x in lst[1:] if x>=pivot])
        return left+[pivot]+right
s = Solution()
ls = s.GetLeastNumbers_Solution([3,5,2,4,1],2)
lst = s.quick_sort([3,5,2,4,1])

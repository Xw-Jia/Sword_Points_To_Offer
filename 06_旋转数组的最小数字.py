'''
非减排序的数组的一个旋转，输出旋转数组的最小元素

二分查找，首元素肯定>=尾元素，找一个中间点，如果它比大的大，说明最小数字再中间点的后面，比小的小，说明最小数字再中间点的前面

'''
class Solution:
    def minNumberInRotateArray(self,nums):
        l, r = 0, len(nums)-1
        if nums[l] < nums[r]:
            return nums[l]
        while l <= r:
            mid = (l+r)//2
            if nums[mid] > nums[l]:
                l = mid
            elif nums[mid] < nums[r]:
                r = mid
            else:
                return nums[r]  #因为右边一直是更小的一方，所以循环完毕，return mid[r]
'''
在长度为n的数组中所有数字都在0-n-1的范围内，有重复。
找出数组中任意一个重复的数字
思路：先排序，再遍历数组查找重复的数字 O(nlgn)
或者建立哈希表，在O(n)查找到
'''
class Solution:
    def duplicate(self, numbers, duplication):
        for i, num in enumerate(numbers):
            while i != num:
                if numbers[num] == num:
                    duplication[0] = numbers[i]
                    return  True
                else:
                    numbers[i], numbers[num] = numbers[num], numbers[i]
                    num = numbers[i]
        return False
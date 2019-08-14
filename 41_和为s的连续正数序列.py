'''
超过target一半的肯定不行，从1，2开始移动指针
'''
class Solution:
    def FindContinuousSequence(self, tsum):
        end = (tsum + 1)//2
        lo, hi, cur_sum = 1, 2, 3
        ans = []
        while lo < end:
            if cur_sum < tsum:
                hi += 1
                cur_sum += hi
            else:
                if cur_sum == tsum:
                    ans.append(list(range(lo, hi+1)))
                cur_sum -= lo
                lo += 1
        return ans
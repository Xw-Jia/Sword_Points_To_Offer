'''

'''
class Solution:
    def reOrderArray(self, array):
        odd , even = [], []
        for i in array:
            odd.append(i) if i%2==1 else even.append(i)
        return odd+even


    #用lambda表达式也可以
    def reOrder(self, array):
        return sorted(array, key = lambda c:c%2, reverse=True)
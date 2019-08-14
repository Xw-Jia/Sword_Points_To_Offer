'''

'''
class Solution:
    def __init__(self):
        self.arr = []
    def Insert(self,num):
        self.arr.append(num)
        self.arr.sort()
    def GetMedian(self, num):
        if len(self.arr)%2 == 1:
            return self.arr[len(self.arr)/2]
        elif len(self.arr)%2 == 0:
            return (self.arr[len(self.arr)/2] + self.arr[len(self.arr)/2-1]) / 2.0
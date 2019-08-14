'''
时间复杂度 O(1)
实现栈的min
思路：建立辅助栈，每次最小值压入辅助栈，辅助栈顶一直就是最小元素，
当数据栈中，最小值被弹出时，同样弹出辅助栈中的栈顶元素
'''
class Solution:
    def __init__(self):
        self.stack = []
        self.minStack = []
    def push(self, node):
        self.stack.append(node)
        if self.minStack==[] or node < self.min():
            self.minStack.append(node)
        else:
            temp = self.min()
            self.minStack.append(temp)
    def pop(self):
        if self.stack==[] or self.minStack==[]:
            return None
        self.minStack.pop()
        self.stack.pop()
    def top(self):
        return self.stack[-1]
    def min(self):
        return self.minStack[-1]
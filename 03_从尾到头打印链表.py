'''
使用栈，用列表模拟
就是使用pop从尾到头打印
'''
class ListNode:
    def __init__(self,x):
        self.val = x
        self.next = None

class Solution:
    def printListFromTailToHead(self, listNode):
        stack = []
        while listNode:
            stack.append(listNode.val)
            listNode = listNode.next
        while stack:
            print(stack.pop())
#方法二：使用递归
    '''
    这个递归真的很有意思
    '''
    def printListFromTail2Head(self, listNode):
        if listNode:
            printListFromTail2Head(listNode.next)
            print(listNode.val)
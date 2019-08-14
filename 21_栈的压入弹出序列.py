'''
输入两个整数序列，第一个序列表示栈的压入顺序，判断第二个是否可能是栈的弹出
'''
class Solution:
    def IsPopOrder(self, pushV, popV):
        if pushV == [] or popV==[]:
            return False
        stack = []
        for i in pushV:
            stack.append(i)
            while len(stack) and stack[-1]==popV[0]:
                stack.pop()
                popV.pop(0) #每次都是辅助栈的栈顶和popV的第一个值判断是否相等，相等就弹出
        if len(stack):
            return False
        else:
            return True

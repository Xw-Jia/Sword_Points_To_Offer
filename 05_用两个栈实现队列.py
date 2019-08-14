'''
一个stack入队，一个出队
出栈为空则从入栈导入道出栈中
push：直接push金stack1，
pop：需要判断stack1和stack2的情况，如果stack2不是空，则直接从stack2中pop，如果stack2为空，把stack1中的值push进stack2中，再popstack2，达到前后反序的目的
'''
class Solution:
    def __init__(self):
        self.stack1 = []
        self.stack2 = []

    def push(self,node):
        self.stack1.append(node)
    def pop(self):
        if len(self.stack1)==0 and len(self.stack2)==0:
            return None
        elif len(self.stack2)==0:
            while len(self.stack1) > 0:
                self.stack2.append(self.stack1.pop())
        return self.stack2.pop()    #所有前面的只是在stack2中反序，return了一个反序后的pop

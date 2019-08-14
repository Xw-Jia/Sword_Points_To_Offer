'''
思路： 引入一个队列，每次打印一个节点的时候，
如果该节点存在子节点，就把该节点的子节点放入队列的末尾，取出队列头部的最早进入队列的节点
'''
class Solution:
    def PrintFromTopToBottom(self, root):
        queue = []
        if not root:
            return []
        result = []
        queue.append(root)
        while len(queue)>0:
            currentRoot = queue.pop(0)
            result.append(currentRoot.val)
            if currentRoot.left:
                queue.append(currentRoot.left)
            if currentRoot.right:
                queue.append(currentRoot.right)
        return result
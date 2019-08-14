'''
把给定的二叉树转换成它的镜像
'''

#使用递归
class Solution:
    def Mirror(self, root):
        if root:
            root.left, root.right = root.right, root.left
            self.Mirror(root.left)
            self.Mirror(root.right)
#使用迭代
class Solution:
    def Mirror(self, root):
        stack = root and [root]
        while stack:
            node = stack.pop()
            if node:
                node.left, node.right = node.right, node.left
                stack += node.right, node.left
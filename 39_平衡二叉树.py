'''
平衡二叉树：空树或者左右两个子树高度差<=1，且子树都是平衡二叉树
递归，在遍历节点时记录深度，一边遍历一边判断
'''
class Solution:
    def __init__(self):
        self.flag = True
    def IsBalanced_Solution(self, pRoot):
        self.getDepth(pRoot)
        return self.flag
    def getDepth(self, root):
        if not root:
            return 0
        left = self.getDepth(root.left) + 1
        right = self.getDepth(root.right) + 1
        if abs(left-right) > 1:
            self.flag = False
        return left if left>right else right
'''
求二叉树的深度
递归，只有一个root，深度为1，存在左子树或右子树，深度为左右子树中深度较深的+1
'''
class Sloution:
    def TreeDepth(self, pRoot):
        if not pRoot:
            return 0
        return max(self.TreeDepth(pRoot.left), self.TreeDepth(pRoot.right)) + 1
'''
输入前序和中序，重建出二叉树并返回

条件：前序遍历的第一个值一定是根节点，对那个中序遍历中间的阶段，再中序的此节点的左侧是左子树，右侧是右子树
使用递归：前序的[0]是root，对应中序的[i];前序的[1:i+1]和中序的[:i]作为对应的左子树继续上一个过程；前序的[i+1:]和中序的[i+1:]对应右子树继续
'''
class Solution:
    def reConstructBinaryTree(self,pre,tin):
        if not pre or not tin:
            return None
        root = TreeNode(pre[0])
        if set(pre) != set(tin):
            return None
        i = tin.index(pre[0])
        root.left = self.reConstructBinaryTree(pre[1:i+1], tin[:i])
        root.right = self.reConstructBinaryTree(pre[i+1:], tin[i+1:])
        return  root
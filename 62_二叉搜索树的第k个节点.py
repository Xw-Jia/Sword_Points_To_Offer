'''
返回第k小的节点,二叉搜索树的中序就是递增排序好的
'''
class Solution:
    def KthNode(self, pRoot, k):
        if not pRoot or not k:
            return None
        res = []
        def traverse(node):
            if len(res) >= k or not node:
                return None
            traverse(node.left)
            res.append(node)
            traverse(node.right)
        traverse(pRoot)
        if len(res) < k:
            return None
        return res[k-1]
'''
输入二叉搜索树，将它转换成一个排序的双向链表，要求不能创建任何新节点，只能调整树中节点指针的指向

思路： 分治，左右子树，递归实现。根节点的左边连接左子树最右边的节点，根节点的右边连接右子树最左边的节点
'''
class Solution:
    def Convert(self, root):
        def convert_tree(node):
            if not node:
                return None
            if node.left:
                left = convert_tree(node.left)
                while left.right:
                    left = left.right
                left.right = node
                node.left = left
            if node.right:
                right = convert_tree(node.right)
                while right.left:
                    right = right.left
                right.left = node
                node.right = right
            return node

        if not root:
            return root
        root = convert_tree(root)
        while root.left:
            root= root.left
        return root
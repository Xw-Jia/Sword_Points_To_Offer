class Solution:
    def Print(self, root):
        ans , level = [], root and []
        while level:
            ans.append([n.val for n in level])
            level = [kid for n in level for kid in (n.left,n.right) if kid]
        return ans
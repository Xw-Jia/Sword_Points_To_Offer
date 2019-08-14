'''
双指针，fast先走k步，然后再一起走
当fast走到尾节点时，slow在倒数第k个基点
'''
class Solution:
    def FindKthToTail(self, head, k):
        fast = slow = head
        for _ in range(k):
            if not fast:
                return None
            fast = fast.next
            while fast:
                slow, fast = slow.next, fast.next
            return slow
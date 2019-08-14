'''
给一个链表，如果包含环，找出链表环的入口节点，否则，输出null

思路：双指针；当fast走到末端，说明没有环；fast==slow,跳出循环
然后head和slow一起走，假设头部走到环a步，环b长，
能够推出 ，相遇时slow走了nb，fast走了2nb，
所以从头再走，head走a，slow走a+nb，相遇，所以都指向入口
'''
class Solution:
    def EntryNodeOfLoop(self, head):
        fast = slow = head
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
            if slow is fast:
                break
        else:
            return None
        while head is not slow:
            head, slow = head.next, slow.next
        return head
'''
需要考虑空链表和只有一个节点的链表
'''
class Solution:
    def ReverseList(self, pHead):
        if not pHead or not pHead.next:
            return pHead
        then = pHead.next
        pHead.next = None
        last = then.next
        while then:
            then.next = pHead
            pHead = then
            then = last
            if then :
                last = then.next
        return  pHead

    #或者，只是简单移动指针
    def reverse(self, pHead):
        prev = None
        while pHead:
            pHead.next, prev, pHead = prev, pHead, pHead.next
        return prev
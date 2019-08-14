'''
2个大王2个小王可以当作任意牌
'''
class Solution:
    def IsContinous(self, numbers):
        if not numbers:
            return False
        joker_count = numbers.count(0)
        left_cards = sorted(numbers)[joker_count:] #剩下的牌中非joker的list
        need_joker = 0
        for i in range(len(left_cards)-1):
            if left_cards[i+1] == left_cards[i]:
                return False
            need_joker += (left_cards[i+1] - left_cards[i] - 1)
        return need_joker <= joker_count
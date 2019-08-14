'''
i am a student --> student a am i

'''
class Solution:
    def ReverseSentence(self, s):
        return ' '.join(reversed(s.split(' ')))

#展开写
class Solution:
    def ReverseSentence(self, s):
        def reverse(s):
            s = s.split(' ')
            for i in range(len(s)//2):
                s[i], s[~i] = s[~i], s[i]
            return ' '.join(s)
        s = reverse(s)
        return s
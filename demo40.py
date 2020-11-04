# demo40 for bayes
spam = .6 * .7 * .1 * .7
non_spam = .3 * .2 * .6 * .3
print(spam, non_spam)

sick = 0.95 * 0.02
health = 0.06 * 0.98
print("sick={},health={}".format(sick, health))
# k*(sick+health)=1 ==> k=1/(sick+health)
k=1/(sick+health)
print("answer={}".format(k*sick))

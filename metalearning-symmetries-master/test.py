
from random import shuffle
import random


l1 = [1,2,3,4,5]
l2 = [1,2,3,4,5]

random.Random().shuffle(l1)
random.Random().shuffle(l2)

print(l1)
print(l2)
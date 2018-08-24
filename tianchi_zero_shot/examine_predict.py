import matplotlib.pyplot as plt
import os
from util import get_dict


nums = 10
all_label_list, name_to_label, label_to_att = get_dict(mode='train')
with open('my_submit.txt') as f:
    names = []
    labels = []
    for i in range(nums):
        x = f.readline()
        x = x.split()
        name = x[0]
        label = x[1]
        names.append(name)
        labels.append(label)

print(names)
print(labels)


print(name_to_label.keys())
for i in range(nums):
    dir1 = os.path.join('data/DatasetA_test_20180813/test/', names[i])
    img = plt.imread(dir1)
    plt.imshow(img)
    title = name_to_label[labels[i]]
    plt.imshow(img)
    plt.title(title)
    plt.show()
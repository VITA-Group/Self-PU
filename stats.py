import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series

f=open('log_4_23//model_30000_1e-3_reverse_dataset.log','r')
a = f.readlines()
s = []
m = 0 
for i in range(2, len(a), 3):
    s.append(set(map(int, a[i][1:-3].split(','))))


for i in range(len(s)):
    if max(s[i]) > m: m = max(s[i])

miou = []
stats = np.zeros(m+1)
early_stats = np.zeros(m+1)
mid_stats = np.zeros(m+1)
end_stats = np.zeros(m+1)

for i in range(len(s) - 1):
    s1 = s[i]
    s2 = s[i + 1]
    sa = s1.union(s2)
    sb = s1.intersection(s2)
    miou.append(len(sb) / len(sa))
    for j in (sa - s1):
        stats[j] += 1
    for j in (sa - s2):
        stats[j] += 1

print("Total Length: {}".format(len(s)))
print("Max: {}".format(max(stats)))
k = max(stats)

counts = []
early_counts = []
mid_counts = []
end_counts = []

while(k > 0):
    print(" = {}, {}".format(k, np.sum(stats == k)))
    counts.extend(np.sum(stats == k) * [k])
    k -= 1
sns.distplot(counts, hist=True)
plt.show()
plt.plot(range(len(miou)), miou)
plt.show()

for i in range(0, 6):
    s1 = s[i]
    s2 = s[i + 1]
    sa = s1.union(s2)
    sb = s1.intersection(s2)
    miou.append(len(sb) / len(sa))
    for j in (sa - s1):
        early_stats[j] += 1
    for j in (sa - s2):
        early_stats[j] += 1
k = 6
while(k > 0):
    print(" = {}, {}".format(k, np.sum(early_stats == k)))
    early_counts.extend(np.sum(early_stats == k) * [k])
    k -= 1
sns.distplot(early_counts, hist=True)
plt.show()

for i in range(6, 12):
    s1 = s[i]
    s2 = s[i + 1]
    sa = s1.union(s2)
    sb = s1.intersection(s2)
    for j in (sa - s1):
        mid_stats[j] += 1
    for j in (sa - s2):
        mid_stats[j] += 1
k = 6
while(k > 0):
    print(" = {}, {}".format(k, np.sum(mid_stats == k)))
    mid_counts.extend(np.sum(mid_stats == k) * [k])
    k -= 1
sns.distplot(mid_counts, hist=True)
plt.show()

for i in range(12, 17):
    s1 = s[i]
    s2 = s[i + 1]
    sa = s1.union(s2)
    sb = s1.intersection(s2)
    for j in (sa - s1):
        end_stats[j] += 1
    for j in (sa - s2):
        end_stats[j] += 1
k = 6
while(k > 0):
    print(" = {}, {}".format(k, np.sum(end_stats == k)))
    end_counts.extend(np.sum(end_stats == k) * [k])
    k -= 1
sns.distplot(end_counts, hist=True)
plt.show()
import json
import math
from matplotlib import pyplot as plt

low = math.log2(1e-2)
high = math.log2(100)
point = math.log2(13)

loss = json.load(open('/Users/qin/workplace/research/examples/word_language_model/wikitext-2_lstm_lr_range_find.json'))

x = [2**(low+(high-low)*i/len(loss)) for i in range(int(len(loss)*(point-low)/(high-low)))]
loss = loss[:int(len(loss)*(point-low)/(high-low))]
plt.plot(x,loss)
plt.xscale('log')
plt.ylabel('loss')
plt.xlabel('lr')
plt.show()
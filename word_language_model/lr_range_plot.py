import json
import math
from matplotlib import pyplot as plt

low = math.log2(1e-4)
high = math.log2(2)

loss = json.load(open('/Users/qin/workplace/research/examples/word_language_model/wikitext-2_lstm_lr_range_find.json'))

x = [2**(low+(high-low)*i/len(loss)) for i in range(int(len(loss)*(high-low)/(high-low)))]
i = loss.index(min(loss))
print(i,2**(low+(high-low)*i/len(loss)))
loss = loss[:int(len(loss)*(high-low)/(high-low))]
plt.plot(x,loss)
plt.xscale('log')
plt.ylabel('loss')
plt.xlabel('lr')
plt.show()
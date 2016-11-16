# plt.hist(train[train.label=='clean'].dwt_1_cV_mean, bins=200, color='b', alpha=0.3)
# plt.hist(train[train.label=='message'].dwt_1_cV_mean, bins=200, color='r', alpha=0.3)
# plt.hist(train[train.label=='clean'].dwt_1_cV_stdev, bins=200, color='b', alpha=0.3)
# plt.hist(train[train.label=='message'].dwt_1_cV_stdev, bins=200, color='r', alpha=0.3)
# plt.hist(train[train.label=='clean'].dwt_1_cV_skew, bins=200, color='b', alpha=0.3)
# plt.hist(train[train.label=='message'].dwt_1_cV_skew, bins=200, color='r', alpha=0.3)
plt.hist(train[train.label=='clean'].dwt_1_cV_kurtosis, bins=200, color='b', alpha=0.3)
plt.hist(train[train.label=='message'].dwt_1_cV_kurtosis, bins=200, color='r', alpha=0.3)
plt.ylim([0, 50])
plt.show()


path = '/home/rokkuran/workspace/stegasawus'

for c in [x for x in train.columns if x not in ('label', 'image')]:
    plt.hist(train[train.label=='clean'][c], bins=200, color='b', alpha=0.3, edgecolor='None')
    plt.hist(train[train.label=='message'][c], bins=200, color='r', alpha=0.3, edgecolor='None')
    plt.ylim([0, 50])
    plt.legend(loc='upper right')
    plt.title(c)
    plt.savefig('{}/output/{}.png'.format(path, c))
    print c
    plt.close()


hist1, bin_edges1 = numpy.histogram(train[train.label=='clean'][c], bins=100)
# plt.bar(bin_edges1[:-1], hist1, color='b', alpha=0.5, edgecolor='None', width=0.5)

hist2, bin_edges2 = numpy.histogram(train[train.label=='message'][c], bins=100)
# plt.bar(bin_edges2[:-1], hist2, color='r', alpha=0.5, edgecolor='None', width=0.5)

hist3 = hist2 - hist1
plt.bar(bin_edges2[:-1], hist3, color='r', alpha=0.5, edgecolor='None', width=0.5)

bin_span = numpy.concatenate((bin_edges1, bin_edges2))
plt.xlim(bin_span.min(), bin_span.max())
plt.show()

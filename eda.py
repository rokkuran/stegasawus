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
# path_train = '{}/data/train_catdog.csv'.format(path)
path_train = '{}/data/train_catdog_ac.csv'.format(path)
train = pandas.read_csv(path_train)

bins = 20
for c in [x for x in train.columns if x not in ('label', 'image')]:
    plt.hist(train[train.label=='cover'][c], bins=bins, color='b', alpha=0.3, edgecolor='None')
    plt.hist(train[train.label=='stego'][c], bins=bins, color='r', alpha=0.3, edgecolor='None')
    # plt.ylim([0, 50])
    plt.legend(loc='upper right')
    plt.title(c)
    plt.savefig('{}/output/{}_bins{}.png'.format(path, c, bins))
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


path = '/home/rokkuran/workspace/stegasawus'
path_train = '{}/data/train_catdog_rndembed_ac.csv'.format(path)
train = pandas.read_csv(path_train)

lc = plt.scatter(
    # train.aca_01[train.label=='cover'],
    train.ac_01_mean[train.label=='cover'],
    train.ac_01_stdev[train.label=='cover'],
    marker='o', color='b', alpha=0.66
)
ls = plt.scatter(
    # train.aca_01[train.label=='stego'],
    train.ac_01_mean[train.label=='stego'],
    train.ac_01_stdev[train.label=='stego'],
    marker='o', color='r', alpha=0.66
)
plt.legend((lc, ls), ('cover', 'stego'))
plt.show()

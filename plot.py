import numpy as np
from matplotlib import pyplot as plt
from scipy import signal

with open('rewards_before/rewardList0_209.npy', 'rb') as f:
    trendList = np.load(f)/3600
    trendStep = (trendList[-1] - trendList[0])/len(trendList)

for i in range(10):

    with open('rewards/rewardList%d_209.npy' % i, 'rb') as f:
        rewards = np.load(f)/3600
        plt.plot(np.cumsum(rewards))
        plt.xlabel("Lépések száma")
        plt.ylabel("Elvesztegetett idő (óra)")
        plt.savefig("rewards-result/%d/reward_learning.png" % i)
        plt.close()

        plt.plot(signal.detrend(np.cumsum(rewards)))
        plt.xlabel("Lépések száma")
        plt.ylabel("Elvesztegetett idő különbség (óra)")
        plt.savefig("rewards-result/%d/reward_learning_det.png" % i)
        plt.close()

        my_detrend = []
        for idx,x in enumerate(rewards):
            my_detrend.append(x - idx*trendStep)

        rewards_detrend_sum = sum(my_detrend)
        print("%d. futas : %d" %(i,rewards_detrend_sum))

    with open('rewards_before/rewardList%d_209.npy'% i, 'rb') as f:
        rewards_before = np.load(f)/3600
        plt.plot(np.cumsum(rewards_before))
        plt.xlabel("Lépések száma")
        plt.ylabel("Elvesztegetett idő (óra)")
        plt.savefig("rewards-result/%d/reward_before.png" %i)
        plt.close()

        plt.plot(signal.detrend(np.cumsum(rewards_before)),alpha=0.6)
        plt.close()

    with open('rewards_after/rewardList%d_209.npy' % i, 'rb') as f:
        rewards_after = np.load(f)/3600
        plt.plot(np.cumsum(rewards_after))
        plt.xlabel("Lépések száma")
        plt.ylabel("Elvesztegetett idő (óra)")
        plt.savefig("rewards-result/%d/reward_after.png" % i)
        plt.close()

        plt.plot(signal.detrend(np.cumsum(rewards_before)), alpha=0.6)
        plt.plot(signal.detrend(np.cumsum(rewards_after)),alpha=0.6)
        plt.legend(["Etalon úthálózat","Javított úthálózat"])
        plt.xlabel("Lépések száma")
        plt.ylabel("Elvesztegetett idő különbség (óra)")

        plt.savefig("rewards-result/%d/reward_before_and_after_det.png" % i)
        plt.close()
        min_index = min(len(np.cumsum(rewards_after)),len(np.cumsum(rewards_before)))
        plt.plot(signal.detrend(np.cumsum(rewards_after))[:min_index]-signal.detrend(np.cumsum(rewards_before))[:min_index])
        plt.xlabel("Lépések száma")
        plt.ylabel("Elvesztegetett idő különbség (óra)")
        plt.savefig("rewards-result/%d/reward_before_after_diff_det.png" % i)
        plt.close()

        my_detrend = []
        for idx, x in enumerate(rewards_after):
            my_detrend.append(x - idx * trendStep)

        rewards_after_detrend_sum = sum(my_detrend)
        print("%d. futas tanulas utan : %f" % (i, rewards_detrend_sum))

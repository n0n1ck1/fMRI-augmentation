import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_metrics(train_losses, val_losses, test_losses, train_f1, test_f1, num_epochs):
    train_losses_np = np.array([lst.tolist() + [0] * (num_epochs - len(lst)) for lst in train_losses])
    test_losses_np = np.array([lst.tolist() + [0] * (num_epochs - len(lst)) for lst in test_losses])
    val_losses_np = np.array([lst.tolist() + [0] * (num_epochs - len(lst)) for lst in val_losses])
    train_f1_np = np.array([lst.tolist() + [0] * (num_epochs - len(lst)) for lst in train_f1])
    test_f1_np = np.array([lst.tolist() + [0] * (num_epochs - len(lst)) for lst in test_f1])

    train_losses_np[train_losses_np == 0] = np.nan
    test_losses_np[test_losses_np == 0] = np.nan
    val_losses_np[val_losses_np == 0] = np.nan
    train_f1_np[train_f1_np == 0] = np.nan
    test_f1_np[test_f1_np == 0] = np.nan

    train_losses_mean = np.nanmean(train_losses_np, axis=0)
    train_losses_std = np.nanstd(train_losses_np, axis=0)
    test_losses_mean = np.nanmean(test_losses_np, axis=0)
    test_losses_std = np.nanstd(test_losses_np, axis=0)
    val_losses_mean = np.nanmean(val_losses_np, axis=0)
    val_losses_std = np.nanstd(val_losses_np, axis=0)
    train_f1_mean = np.nanmean(train_f1_np, axis=0)
    train_f1_std = np.nanstd(train_f1_np, axis=0)
    test_f1_mean = np.nanmean(test_f1_np, axis=0)
    test_f1_std = np.nanstd(test_f1_np, axis=0)

    plt.figure(figsize=(8, 6))
    plt.errorbar(range(1, 1 + train_losses_mean.shape[0]), train_losses_mean, yerr=train_losses_std, fmt='-o', label='Train losses')
    plt.errorbar(range(1, 1 + test_losses_mean.shape[0]), test_losses_mean, yerr=test_losses_std, fmt='-o', label='Test losses')
    plt.errorbar(range(1, 1 + val_losses_mean.shape[0]), val_losses_mean, yerr=val_losses_std, fmt='-o', label='Val losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Cross entropy loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.errorbar(range(1, 1 + train_f1_mean.shape[0]), train_f1_mean, yerr=train_f1_std, fmt='-o', label='Train F1')
    plt.errorbar(range(1, 1 + test_f1_mean.shape[0]), test_f1_mean, yerr=test_f1_std, fmt='-o', label='Test F1')
    plt.xlabel('Epoch')
    plt.ylabel('F1 score')
    plt.title('F1 score')
    plt.legend()
    plt.grid(True)
    plt.show()

    return test_f1_np

def plot_aug(real, aug, sub, net):
    print(np.corrcoef(aug[sub, :, net], real[sub, :, net])[0, 1])
    plt.figure(figsize=(15, 5))
    plt.plot(real[sub, :, net], label='real')
    plt.plot(aug[sub, :, net], label='aug')
    plt.legend()
    plt.title(f'Augmented data');
    plt.show()
    
def plot(real, subs, nets, colors, title):
    plt.figure(figsize=(15, 5))
    for i in range(subs.shape[0]):
        plt.plot(real[subs[i], :, int(nets[i])], label=str(subs[i]), color=colors[i])
    plt.legend()
    plt.title(title)
    plt.show()
    
def get_stats(values):
    mean = np.nanmean(values, axis=0)
    std = np.nanstd(values, axis=0)
    return np.max(mean), std[np.argmax(mean)]

def plot_results(new_real_f1, new_aug_f1, new_comb_f1, old_real_f1, old_aug_f1, old_comb_f1):
    new_real_f1_mean, new_real_f1_std = get_stats(new_real_f1)
    new_aug_f1_mean, new_aug_f1_std = get_stats(new_aug_f1)
    new_comb_f1_mean, new_comb_f1_std = get_stats(new_comb_f1)

    old_real_f1_mean, old_real_f1_std = get_stats(old_real_f1)
    old_aug_f1_mean, old_aug_f1_std = get_stats(old_aug_f1)
    old_comb_f1_mean, old_comb_f1_std = get_stats(old_comb_f1)
    
    f1_means = [new_real_f1_mean, new_aug_f1_mean, new_comb_f1_mean,
                 old_real_f1_mean, old_aug_f1_mean, old_comb_f1_mean]

    f1_stds = [new_real_f1_std, new_aug_f1_std, new_comb_f1_std,
                old_real_f1_std, old_aug_f1_std, old_comb_f1_std]

    text = ["new real", "new augmented", "new combined",
            "old real", "old augmented", "old combined"]
    
    plt.figure(figsize=(8, 6))
    plt.errorbar(text, f1_means, yerr=f1_stds)
    plt.xticks(rotation=30)
    plt.xlabel('Training dataset')
    plt.ylabel('F1 score')
    plt.title('Test F1 score')
    plt.grid(True)
    plt.show()
    
    new_real_f1 = np.nanmax(new_real_f1, axis=1)
    new_aug_f1 = np.nanmax(new_aug_f1, axis=1)
    new_comb_f1 = np.nanmax(new_comb_f1, axis=1)
    old_real_f1 = np.nanmax(old_real_f1, axis=1)
    old_aug_f1 = np.nanmax(old_aug_f1, axis=1)
    old_comb_f1 = np.nanmax(old_comb_f1, axis=1)
    
    combined_data = np.concatenate([new_real_f1, new_aug_f1, new_comb_f1,
                                    old_real_f1, old_aug_f1, old_comb_f1])
    labels = np.repeat(text, 15)
    sns.boxplot(x=labels, y=combined_data)
    plt.xticks(rotation=30)
    plt.title('Test F1 score')
    plt.xlabel('Training dataset')
    plt.ylabel('F1 score')
    plt.show()
    
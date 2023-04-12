import os
import shutil
import random
import torch

def split_dataset(images_dir, train_dir, test_dir, train_ratio, seed):
    if os.path.exists(train_dir):
        shutil.rmtree(train_dir)
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

    classes = os.listdir(images_dir)

    os.makedirs(train_dir)
    os.makedirs(test_dir)

    for c in classes:
        class_dir = os.path.join(images_dir, c)

        images = os.listdir(class_dir)
        random.Random(seed).shuffle(images)

        n_train = int(len(images) * train_ratio)

        train_images = images[:n_train]
        test_images = images[n_train:]

        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(test_dir, c), exist_ok=True)

        for image in train_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(train_dir, c, image)
            shutil.copyfile(image_src, image_dst)

        for image in test_images:
            image_src = os.path.join(class_dir, image)
            image_dst = os.path.join(test_dir, c, image)
            shutil.copyfile(image_src, image_dst)

def get_dataset_mean_std(dataset):
    means = torch.zeros(3)
    stds = torch.zeros(3)

    for img, label in dataset:
        means += torch.mean(img, dim=(1, 2))
        stds += torch.std(img, dim=(1, 2))

    means /= len(dataset)
    stds /= len(dataset)

    print(f'Calculated means: {means}')
    print(f'Calculated stds: {stds}')

    return means, stds
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as F

def normalize_image(image):
    """Normalize the given image by clamping it between its minimum and maximum values and scaling it."""
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image  

def plot_images(images, labels, classes, normalize=True):
    """Plot a grid of images with their corresponding labels.

    Args:
        images (torch.Tensor): a tensor containing the images to plot.
        labels (torch.Tensor): a tensor containing the labels corresponding to each image.
        classes (list): a list of integers representing the class indices of each label.
        normalize (bool, optional): whether to normalize the images or not. Defaults to True.
    """

    classes_names = {
        '0':'No DR',
        '1':'Mild',
        '2':'Moderate',
        '3':'Severe',
        '4':'Proliferative DR'
    }

    n_images = len(images)

    rows = int(np.sqrt(n_images))
    cols = int(np.sqrt(n_images))

    fig = plt.figure(figsize=(15, 15))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i+1)
        
        image = images[i]

        if normalize:
            image = normalize_image(image)

        ax.imshow(image.permute(1, 2, 0).cpu().numpy())
        label = classes_names[classes[labels[i]]]
        ax.set_title(label)
        ax.axis('off')
    fig.show()

def plot_class_distribution(dataset, class_names):
    """Plot the class distribution of a dataset.

    Args:
        dataset (torch.utils.data.Dataset): the dataset to plot.
        class_names (list): a list of integers representing the class indices to plot.
    """

    classes_names = {
        '0':'No DR',
        '1':'Mild',
        '2':'Moderate',
        '3':'Severe',
        '4':'Proliferative DR'
    }

    class_names = [classes_names[i] for i in class_names]

    class_counts = [0] * len(class_names)
    for _, label in dataset:
        class_counts[label] += 1

    fig, ax = plt.subplots()
    sns.barplot(x=class_names, y=class_counts, ax=ax)
    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Images')
    ax.set_title('Class Distribution of Dataset')
    plt.show()


def plot_transformations(sample_image, pretrained_size, pretrained_means, pretrained_stds):

    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    axs = axs.ravel()

    # Plot the original image
    axs[0].imshow(np.transpose(sample_image, (1, 2, 0)))
    axs[0].set_title("Original")

    # Define the resize transformation
    resize_transform = transforms.Resize(pretrained_size)

    # Apply the resize transformation to the sample image
    resized_image = resize_transform(F.to_pil_image(sample_image))

    # Plot the resized image
    axs[1].imshow(resized_image)
    axs[1].set_title("Resized")

    # Define the rotation transformation
    rotation_transform = transforms.RandomRotation(5)

    # Apply the rotation transformation to the sample image
    rotated_image = rotation_transform(F.to_pil_image(sample_image))

    # Plot the rotated image
    axs[2].imshow(rotated_image)
    axs[2].set_title("Rotated")

    # Define the horizontal flip transformation
    hflip_transform = transforms.RandomHorizontalFlip(0.5)

    # Apply the horizontal flip transformation to the sample image
    hflipped_image = hflip_transform(F.to_pil_image(sample_image))

    # Plot the horizontally flipped image
    axs[3].imshow(hflipped_image)
    axs[3].set_title("Horizontally Flipped")

    # Define the random crop transformation
    crop_transform = transforms.RandomCrop(pretrained_size, padding=10)

    # Apply the random crop transformation to the sample image
    cropped_image = crop_transform(F.to_pil_image(sample_image))

    # Plot the cropped image
    axs[4].imshow(cropped_image)
    axs[4].set_title("Cropped")

    axs[5].axis('off')

    # Set the spacing between subplots
    fig.tight_layout(pad=2.0)

    # Show the plot
    plt.show()







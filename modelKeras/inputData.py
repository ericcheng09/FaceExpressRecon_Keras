import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
import os

pic_size = 48

path = "trainingData/images/"


def getData(batch_size=128):
    """
    Parameters:
        batch_size

    Returns:
            Two ImageDataGenerator object (train, validation)

    """

    train_Datagen = ImageDataGenerator()
    valid_Datagen = ImageDataGenerator()

    train_gen = train_Datagen.flow_from_directory(path + "train",
                                                target_size = (pic_size, pic_size),
                                                color_mode="grayscale",
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=True)
    valid_gen = valid_Datagen.flow_from_directory(path + "validation",
                                                target_size = (pic_size, pic_size),
                                                color_mode="grayscale",
                                                batch_size=batch_size,
                                                class_mode='categorical',
                                                shuffle=False)


    return train_gen, valid_gen


if __name__ == "__main__":
    plt.figure(0, figsize=(12, 20))
    idx = 0

    for expression in os.listdir(path + "train"):
        for i in range(5):
            idx += 1
            plt.subplot(7, 5, idx)
            img = load_img(path + "train/" + expression + "/" +os.listdir(path + "train/" + expression)[i], target_size=(pic_size, pic_size))

            plt.imshow(img, cmap="gray")
    for expression in os.listdir(path + "train"):
        print(str(len(os.listdir(path + "train/" + expression))) + " " + expression + " images")
    plt.show()



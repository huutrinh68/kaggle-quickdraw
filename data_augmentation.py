from common import *
""" From URL: https://keras.io/preprocessing/image/
"""

datagen = ImageDataGenerator(
    rotation_range=90,      # can set from -90 ~ 90
    fill_mode='nearest',    # can be one of these value: constant,nearest,reflect,wrap
    width_shift_range=0.5,  # can be int, 1-D array, float
    height_shift_range=0.2, # can be int, 1-D array, float
    channel_shift_range=50, # random change color value in range(50)
    shear_range=5,          # dont set it too big, It makes generated data isn't relevant with original data.
    zoom_range=[1, 1.5],    # int: zoom in range[lower, upper], float: zoom in range[1-zoom_range, 1+zoom_range]    
    horizontal_flip=True,
    vertical_flip=True
)

def show_imgs(imgs, row, col):
    """ Show PILimage as row*col
        # Arguments
        :imgs: 1-D images, include PILimages
        :row: Int, row for plt.subplot
        :col: Int, column for plt.subplot
    """
    if len(imgs) != (row*col):
        raise ValueError("Invalid imgs len:{} col:{} row:{}".format(len(imgs), row, col))
    
    for i, img in enumerate(imgs):
        plot_num = i+1
        plt.subplot(row, col, plot_num)
        plt.tick_params(labelbottom=False)  # remove x axis
        plt.tick_params(labelleft=False)    # remove y axis
        plt.imshow(img)
    plt.show()

def main():
    # files = os.listdir(os.path.join(root_path, 'data'))
    img_path = os.path.join(root_path, 'data/lena.jpeg')
    saved_folder = os.path.join(root_path, 'data/augmented_data')
    os.makedirs(saved_folder, exist_ok=True)

    # input image, Pil format
    img = image.load_img(img_path, color_mode='rgb', target_size=None, interpolation='nearest')

    # pil format to array format
    array_data = image.img_to_array(img)
    
    #(height, width, 3) -> (1, height, width, 3)
    array_data = array_data.reshape((1,) + array_data.shape)

    max_img_num = 16
    imgs = []

    for i, d in enumerate(datagen.flow(array_data, batch_size=1)):
        # array format to pil format
        img = image.array_to_img(d[0], scale=True)

        # then save to saved_folder
        img_name = os.path.join(saved_folder, str(i+1) + '.png')
        image.save_img(img_name, img)
        imgs.append(img)
        if (len(imgs) % max_img_num) == 0:
            break

    # show_imgs(imgs, row=4, col=4)


if __name__ == '__main__':
    main()
    
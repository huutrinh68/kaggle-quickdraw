from common import *
""" From URL: https://keras.io/preprocessing/image/
"""

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


img_path = 'lena.jpeg'
# input image, Pil format
img = image.load_img(img_path)
# pil format to array format
x = image.img_to_array(img)
#(h, w, 3) -> (1, h, w, 3)
x = x.reshape((1,) + x.shape)

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

max_img_num = 16
imgs = []

for d in datagen.flow(x, batch_size=1):
    # array format to pil format 
    imgs.append(image.array_to_img(d[0], scale=True))
    if len(imgs) % max_img_num == 0:
        break

show_imgs(imgs, row=4, col=4)

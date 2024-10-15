import glob
import cv2
import pathlib


from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array, load_img

IMAGES = []
DATASET_PATH= "./data/shelf"
IMAGES_PATH = glob.glob(f"{DATASET_PATH}/*.png")



datagen = ImageDataGenerator(
    rotation_range=3,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest')

# # load data
# for img_path in IMAGES_PATH:
#     img = cv2.imread(img_path)
#     IMAGES.append(img)
#
# #
# i = 0
# for img in IMAGES:

#     img = load_img(path)
#     x = img_to_array(img)
#     x = x.reshape((1,) + x.shape)
#
#     for batch in datagen.flow(x,batch_size=1,save_to_dir='preview',save_prefix='shelf',save_format='png'):
#        i += 1
#        if i>20:
#            break
name = 'IMG_1427'
path = f'./data/shelf/{name}.png'
img = load_img(path)
x = img_to_array(img)
x = x.reshape((1,) + x.shape)

i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir='preview', save_prefix=name, save_format='png'):
   i += 1
   if i>20:
       break

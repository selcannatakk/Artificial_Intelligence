import os
from moviepy.editor import ImageSequenceClip


'''

images_folder = videoyu oluşturacak resimlerin klosör konumu

'''


images_folder = ''


# image_files = [images_folder + img for img in sorted(os.listdir(images_folder)) if img.endswith(".jpg")]
image_files = []
for img in sorted(os.listdir(images_folder)):
    if img.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        image_files.append(os.path.join(images_folder, img))

# Videoyu oluşturmak için resim dosyalarını kullanarak ImageSequenceClip oluşturuyoruz
# fps : saniyedeki kare sayısı
clip = ImageSequenceClip(image_files, fps=1)

# saving
clip.write_videofile("./output_data/license_plate_video.mp4")

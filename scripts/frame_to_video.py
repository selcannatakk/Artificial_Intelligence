import cv2
import os
import time

# Görsellerin bulunduğu klasör
# images_dir = "C:/Users/selca/selcanatak/.artificial-intelligence/ai_projects/metro_safe_track/results/1/images/"
images_dir = "C:/Users/selca/selcanatak/.artificial-intelligence/ai_projects/metro_safe_track/result_video_images/"
# Video parametreleri
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter("create_video2.mp4", fourcc, 20.0, (640, 480))

end_time = time.time()

# Dosyaları sıralı olarak listelemek için `sorted()` kullanıyoruz

i = 1
for i in range(len(os.listdir(images_dir))):

    frame_path = os.path.join(images_dir, f"frame{i}.png")
    frame = cv2.imread(frame_path)

    # Dosyanın başarıyla okunup okunmadığını kontrol edin
    if frame is None:
        print(f"'{frame_path}' dosyası okunamadı!")
        continue

    # Görseli RGB'ye çevir ve yeniden boyutlandır
    # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (640, 480))

    # FPS hesaplama (isteğe bağlı)
    start_time = time.time()
    fps = 1 / (start_time - end_time) if end_time != 0 else 0
    end_time = start_time

    # Görüntüyü videoya ekleme
    video_writer.write(frame)
    i+=1

# Kaynakları serbest bırakma
video_writer.release()
cv2.destroyAllWindows()

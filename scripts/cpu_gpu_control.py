import torch
import tensorflow as tf

print("------ PyTorch CUDA Bilgileri ------")
if torch.cuda.is_available():
    print("PyTorch CUDA Desteği: Evet")
    print("CUDA Sürümü:", torch.version.cuda)
    print("PyTorch CUDA Sürümü:", torch.backends.cudnn.version())

    # GPU özellikleri
    gpu_count = torch.cuda.device_count()
    print("Toplam GPU Sayısı:", gpu_count)

    for i in range(gpu_count):
        print(f"--- GPU {i} Bilgileri ---")
        print("Cihaz Adı:", torch.cuda.get_device_name(i))
        print("Toplam Bellek (MB):", torch.cuda.get_device_properties(i).total_memory // 1024 ** 2)
        print("Cihaz İndeksi:", torch.cuda.current_device())
else:
    print("PyTorch CUDA Desteği: Hayır")




print("\n------ TensorFlow CUDA Bilgileri ------")
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("TensorFlow CUDA Desteği: Evet")
    print("Toplam GPU Sayısı:", len(gpus))

    for gpu in gpus:
        print(f"--- GPU Cihaz Bilgisi: {gpu} ---")
        details = tf.config.experimental.get_device_details(gpu)
        print("Cihaz Adı:", details.get("device_name", "Bilinmiyor"))
        print("Compute Capability:", details.get("compute_capability", "Bilinmiyor"))
else:
    print("TensorFlow CUDA Desteği: Hayır")

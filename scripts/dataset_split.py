import os
import random
import shutil


def copy_files(dataset_path, dataset, dataset_type):

    os.makedirs(dataset_type+"/images", exist_ok=True)
    os.makedirs(dataset_type+"/labels", exist_ok=True)

    for img, txt in dataset:
        shutil.copy(os.path.join(dataset_path,"images", img), os.path.join(dataset_type,"images", img))
        shutil.copy(os.path.join(dataset_path,"labels", txt), os.path.join(dataset_type,"labels", txt))


    print("Dosyalar başarıyla ayrıldı!")


def dataset_split(matched_dataset,train_size,val_size):
    random.shuffle(matched_dataset)

    matched_dataset_count = len(matched_dataset)

    train_count = int(matched_dataset_count * train_size)
    val_count = int(matched_dataset_count * val_size)

    train_dataset = matched_dataset[:train_count]
    val_dataset = matched_dataset[train_count:train_count + val_count]
    test_dataset = matched_dataset[train_count + val_count:]


    return train_dataset,val_dataset,test_dataset

def main():
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15


    dataset_path = "./dataset/license_plate_yolo_od"

    images_path =os.path.join(dataset_path,"images")
    annotations_path =os.path.join(dataset_path,"labels")

    train_dir = "./dataset/license_plate_yolo_od/train"
    val_dir = "./dataset/license_plate_yolo_od/val"
    test_dir = "./dataset/license_plate_yolo_od/test"

    for dir_name in [train_dir, val_dir, test_dir]:
        os.makedirs(dir_name, exist_ok=True)


    image_files = [f for f in os.listdir(images_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    annotation_files = [f for f in os.listdir(annotations_path) if f.lower().endswith('.txt')]



    matched_dataset = []
    for image_file in image_files:

        name, ext = image_file.split('.')

        annotation_file = os.path.join(name+".txt")

        if annotation_file in annotation_files:

            matched_dataset.append((image_file, annotation_file))


    train_dataset,val_dataset,test_dataset = dataset_split(matched_dataset,train_size,val_size)


    copy_files(dataset_path,train_dataset,train_dir)
    copy_files(dataset_path,val_dataset,val_dir)
    copy_files(dataset_path,test_dataset,test_dir)


if __name__ == "__main__":
    main()
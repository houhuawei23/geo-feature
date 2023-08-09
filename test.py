import os
import shutil

# Paths for the original and the small dataset
cur_dir = os.getcwd()
datas_dir = os.path.join(cur_dir, 'datas')
original_dataset_dir = os.path.join(datas_dir, 'mnist', 'train')
small_dataset_dir = os.path.join(datas_dir, 'mnist_10', 'train')

# Create the small dataset directory if it doesn't exist
os.makedirs(small_dataset_dir, exist_ok=True)

# Loop over the classes directories

for class_dir in os.listdir(original_dataset_dir):
    # if class_dir not number
    if class_dir.isdigit() == False:
        continue
    if int(class_dir) == 0:
        print("here is 0")
    # Create the class directory in the small dataset directory
    os.makedirs(os.path.join(small_dataset_dir, class_dir), exist_ok=True)

    # Get a list of all the image filenames in the class directory
    image_filenames = os.listdir(os.path.join(original_dataset_dir, class_dir))
    # sort
    image_filenames.sort(key=lambda x: int(x[:-4]))
    # image_filenames.sort()
    # Only copy the first 200 images
    for filename in image_filenames[:10]:
        # Construct the full path for the original image and the new image
        src = os.path.join(original_dataset_dir, class_dir, filename)
        dst = os.path.join(small_dataset_dir, class_dir, filename)
        # print(src, dst)
        # Copy the image
        shutil.copyfile(src, dst)

print('Done')
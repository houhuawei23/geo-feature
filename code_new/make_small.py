import os
import shutil

dataset_name = 'mnist'
num_perclass = 10

# Paths for the original and the small dataset
curfile_dir = os.path.dirname(os.path.abspath(__file__))
dataset_dir = os.path.join(curfile_dir, '..', 'dataset')
original_dataset_dir = os.path.join(dataset_dir, dataset_name, 'train')
small_dataset_dir = os.path.join(dataset_dir, '{0}_{1}'.format(dataset_name, str(num_perclass)), 'train')
# print(curfile_dir, datas_dir, original_dataset_dir, small_dataset_dir)
# Create the small dataset directory if it doesn't exist
os.makedirs(small_dataset_dir, exist_ok=True)

# Loop over the classes directories

for class_dir in os.listdir(original_dataset_dir):
    # if class_dir not number
    if class_dir.isdigit() == False:
        continue
    # Create the class directory in the small dataset directory
    os.makedirs(os.path.join(small_dataset_dir, class_dir), exist_ok=True)

    # Get a list of all the image filenames in the class directory
    image_filenames = os.listdir(os.path.join(original_dataset_dir, class_dir))
    # sort
    image_filenames.sort(key=lambda x: int(x[:-4]))
    # image_filenames.sort()
    # Only copy the first 200 images
    num_imges = min(num_perclass, len(image_filenames))  
    for filename in image_filenames[:num_imges]:
        # Construct the full path for the original image and the new image
        src = os.path.join(original_dataset_dir, class_dir, filename)
        dst = os.path.join(small_dataset_dir, class_dir, filename)
        # print(src, dst)
        # Copy the image
        shutil.copyfile(src, dst)

print('make_small.py done, num_perclass =', num_perclass)
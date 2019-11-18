from PIL import Image
import csv
from collections import defaultdict
import random as rand
import os



LISA_DOWNLOAD_PATH = '/Users/timmcdermott/Downloads/signDatabasePublicFramesOnly/'
LISA_NEW_DATA_PATH = '/Users/timmcdermott/Documents/CSCE482/lisaData/'
ANNOTATIONS_FILE_PATH = LISA_DOWNLOAD_PATH + "allAnnotations.csv"
NEGATIVE_FILE_PATH = LISA_DOWNLOAD_PATH + 'negatives/negativePics/'
TRAIN_PERC = .8
TEST_PERC = 1-TRAIN_PERC
OUTPUT_IMG_EXTENSION = '.jpg'
RESIZE_TUPLE = (416, 416)


# Saves the image received in the output file path in the OUTPUT_IMG_EXTENSION.
# Saves the filename in the general training/testing file.
# Saves the filename and darknet labels for each object in the txt file with the image filename.
def write_data(filename, input_img, darknet_format, text_file, output_dir):
    output_file_path = output_dir + filename

    # Save file in general training/testing file
    text_file.write(output_file_path + OUTPUT_IMG_EXTENSION + "\n")
    # Save file in correct folder with new extension
    input_img.save(output_file_path + OUTPUT_IMG_EXTENSION)

    # SAVE TXT FILE
    with open(output_file_path + '.txt', "a+") as f:
        f.write(darknet_format)

def calculate_darknet_format(input_img, row, labelNums):
    real_img_width, real_img_height = input_img.size
    image_width = int(real_img_width)
    image_height = int(real_img_height)

    left_x = float(row[2])# / image_width
    bottom_y = float(row[3])# / image_height
    right_x = float(row[4])# / image_width
    top_y = float(row[5])# / image_height

    object_class = labelNums[row[1]]

    # print(object_class, image_width, image_height, left_x, right_x, bottom_y, top_y)

    return parse_darknet_format(object_class, image_width, image_height, left_x, bottom_y, right_x, top_y)


def parse_darknet_format(object_class, img_width, img_height, left_x, bottom_y, right_x, top_y):
    object_width = right_x - left_x
    object_height = top_y - bottom_y
    object_mid_x = (left_x + right_x) / 2.0
    object_mid_y = (bottom_y + top_y) / 2.0

    # print(object_width, object_height, object_mid_x, object_mid_y, img_height, img_width)

    object_width_rel = object_width / img_width
    object_height_rel = object_height / img_height
    object_mid_x_rel = object_mid_x / img_width
    object_mid_y_rel = object_mid_y / img_height

    dark_net_label = "{0} {1:.10f} {2:.10f} {3:.10f} {4:.10f}". \
        format(object_class, object_mid_x_rel, object_mid_y_rel, object_width_rel, object_height_rel)

    return dark_net_label


anFile = open(ANNOTATIONS_FILE_PATH)  # Annotations file
gtReader = csv.reader(anFile, delimiter=';')
train_text_file = open(LISA_NEW_DATA_PATH + 'train.txt', "w+")
test_text_file = open(LISA_NEW_DATA_PATH + 'test.txt', "w+")
output_train_dir_path = LISA_NEW_DATA_PATH + 'train/'
output_test_dir_path = LISA_NEW_DATA_PATH + 'test/'

labels = defaultdict(list)
num_train_files, num_test_files = 0, 0

# Get all annotations
for line in gtReader:
    if line[0].split("/")[-1][-4:] != '.png': continue # Ignore video annotations

    filename = line[0].split("/")[-1][:-4]
    file_path = LISA_DOWNLOAD_PATH + line[0]
    label = line[1]

    labels[label].append((file_path, line))

# Labels must be integers in annotations, so create this map here
labelNums = {label:i for i, label in enumerate(labels.keys())}

for imageList in labels.values():
    for imageTuple in imageList:
        image = Image.open(imageTuple[0])
        line = imageTuple[1]

        darknet_format = calculate_darknet_format(image, line, labelNums)

        train_file = rand.choices([True, False], [TRAIN_PERC, TEST_PERC])[0]

        if train_file:
            num_train_files += 1
            write_data(filename, image, darknet_format, train_text_file, output_train_dir_path)
        else:
            num_test_files += 1
            write_data(filename, image, darknet_format, test_text_file, output_test_dir_path)

print("BEFORE ADDING NEGATIVES -- Num train files: " + str(num_train_files) + ', num test files: ' + str(num_test_files))

# Write all labels to lisa.names
with open(LISA_NEW_DATA_PATH + 'lisa.names', 'w+') as f:
    for label, num in sorted(labelNums.items(), key=lambda x:x[1]):
        if num > 0:
            f.write('\n')
        f.write(label)

# Move negatives (images without labels) to test and train
for root, dirs, files in os.walk(NEGATIVE_FILE_PATH):
    for file in files:
        # try:
        if file.endswith(".png"):
            image = Image.open(os.path.join(root, file))
            # print(file)
            # Resize image
            # resized_img = image.resize(RESIZE_TUPLE)
            train_file = rand.choices([True, False], [TRAIN_PERC, TEST_PERC])[0]
            if train_file:
                num_train_files += 1
                # train_text_file.write(LISA_NEW_DATA_PATH + file.replace('.png', '.jpg'))
                # image.save(os.path.join(output_train_dir_path, file.replace('.png', '.jpg')))
                write_data(file[:-4], image, '', train_text_file, output_train_dir_path)
            else:
                num_test_files += 1
                # test_text_file.write(LISA_NEW_DATA_PATH + file.replace('.png', '.jpg'))
                # image.save(os.path.join(output_train_dir_path, file.replace('.png', '.jpg')))
                write_data(file[:-4], image, '', test_text_file, output_test_dir_path)
        # except:
            # continue

print('number of labels: ')
print(len(labels.keys()))
print(labels)

print("AFTER ADDING NEGATIVES -- Num train files: " + str(num_train_files) + ', num test files: ' + str(num_test_files))

anFile.close()
test_text_file.close()
train_text_file.close()
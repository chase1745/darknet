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

def calculate_darknet_format(input_img, row):
    real_img_width, real_img_height = input_img.size
    image_width = int(real_img_width)
    image_height = int(real_img_height)

    left_x = float(row[2]) / image_width
    bottom_y = float(row[3]) / image_height
    right_x = float(row[4]) / image_width
    top_y = float(row[5]) / image_height

    object_class = row[1]

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

    dark_net_label = "{} {} {} {} {}". \
        format(object_class, object_mid_x_rel, object_mid_y_rel, object_width_rel, object_height_rel)

    return dark_net_label


anFile = open(ANNOTATIONS_FILE_PATH)  # Annotations file
gtReader = csv.reader(anFile, delimiter=';')
train_text_file = open(LISA_NEW_DATA_PATH + 'train.txt', "a+")
test_text_file = open(LISA_NEW_DATA_PATH + 'test.txt', "a+")
output_train_dir_path = LISA_NEW_DATA_PATH + 'train/'
output_test_dir_path = LISA_NEW_DATA_PATH + 'test/'

labels = defaultdict(int)
num_train_files, num_test_files = 0, 0

# Get all annotations
for line in gtReader:
    if line[0].split("/")[-1][-4:] != '.png': continue # Ignore video annotations

    filename = line[0].split("/")[-1][:-4]
    file_path = LISA_DOWNLOAD_PATH + line[0]
    label = line[1]
    labels[label] += 1

    image = Image.open(file_path)
    darknet_format = calculate_darknet_format(image, line)
    print(darknet_format)

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
    for label in labels.keys():
        f.write(label + '\n')

# Move negatives (images without labels) to test and train
for root, dirs, files in os.walk(NEGATIVE_FILE_PATH):
    for file in files:
        try:
            if file.endswith(".png"):
                image = Image.open(os.path.join(root, file))
                train_file = rand.choices([True, False], [TRAIN_PERC, TEST_PERC])[0]
                if train_file:
                    num_train_files += 1
                    train_text_file.write(LISA_NEW_DATA_PATH + file.replace('.png', '.jpg'))
                    image.save(os.path.join(output_train_dir_path, file.replace('.png', '.jpg')))

                else:
                    num_test_files += 1
                    test_text_file.write(LISA_NEW_DATA_PATH + file.replace('.png', '.jpg'))
                    image.save(os.path.join(output_train_dir_path, file.replace('.png', '.jpg')))
        except:
            continue

print('labels: ')
print(labels)

print("AFTER ADDING NEGATIVES -- Num train files: " + str(num_train_files) + ', num test files: ' + str(num_test_files))

anFile.close()
test_text_file.close()
train_text_file.close()
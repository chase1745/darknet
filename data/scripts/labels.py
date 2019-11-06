from PIL import Image

with open('../tsr_data/gt.txt', 'r') as gt:
    for line in gt.readlines():
        fname, l, u, r, d, classNum  = line.rstrip().split(';')
        l, u, r, d = int(l), int(u), int(r), int(d)
        with open('../tsr_data/' + fname, mode='rb') as imageFile:
            imageFormat = imageFile.readline()
            _, w, h, _ = imageFormat.split()
            imageW, imageH = int(w), int(h)
            boxW = (r-l)/imageW
            boxH = (d-u)/imageH
            x = l/imageW
            y = d/imageH

            with open('../tsr_data/' + fname.replace('ppm', 'txt'), 'a+') as labelFile:
                labelFile.write('../tsr_data/' + classNum + ' ' + str(x) + ' ' + str(y) + ' ' + str(boxW) + ' ' + str(boxH) + '\n')

            im = Image.open('../tsr_data/' + fname)
            im.save('../tsr_data/' + fname.replace('ppm', 'jpg'))
from PIL import Image

def convert(size, box):
    dw = 1./(size[0])
    dh = 1./(size[1])
    x = (box[0] + box[1])/2.0 - 1
    y = (box[2] + box[3])/2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

with open('../tsr_data/gt.txt', 'r') as gt:
    for line in gt.readlines():
        fname, l, d, r, u, classNum  = line.rstrip().split(';')
        l, u, r, d = int(l), int(u), int(r), int(d)
        with open('../tsr_data/' + fname, mode='rb') as imageFile:
            imageFormat = imageFile.readline()
            _, w, h, _ = imageFormat.split()
            imageW, imageH = int(w), int(h)
            x, y, boxW, boxH = convert([imageW, imageH], [r, l, u, d])
            # boxW = (r-l)/imageW
            # boxH = (d-u)/imageH
            # x = ((r+l)/2)/imageW
            # y = ((d+u)/2)/imageH

            with open('../tsr_data/' + fname.replace('ppm', 'txt'), 'a+') as labelFile:
                labelFile.write('../tsr_data/' + classNum + ' ' + str(x) + ' ' + str(y) + ' ' + str(boxW) + ' ' + str(boxH) + '\n')

            im = Image.open('../tsr_data/' + fname)
            im.save('../tsr_data/' + fname.replace('ppm', 'jpg'))
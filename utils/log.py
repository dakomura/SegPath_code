import cv2
import datetime
import numpy as np
import os
import matplotlib.pyplot as plt


class Log:
    def __init__(self, op, sample, imgsize, out_scale):
        self.areas = []
        self.blurs = []
        self.op = op
        self.sample = sample
        self.scale = 6
        self.datetime = datetime.datetime.now()
        self.otsu_th = None

        self.imgsize = int(imgsize * (2 ** (out_scale - self.scale)))

        w, h = self.op.dimensions
        self.img = op.read_region([0, 0], self.scale,
                                  [int(w / (2 ** self.scale)),
                                   int(h / (2 ** self.scale))])
        self.img = np.array(self.img)

    def add(self, x, y, pr, blur):
        x2 = int(x / (2 ** self.scale))
        y2 = int(y / (2 ** self.scale))
        self.areas.append(float(pr) / 100.0)

        if float(pr) > 0:
            self.blurs.append(float(blur))

        self.img = cv2.rectangle(self.img, (x2, y2),
                                 (x2 + self.imgsize, y2 + self.imgsize),
                                 (0, 255, 255), 3)

    def add_otsu(self, th):
        self.otsu_th = th

    def plot(self, contour_img, outdir):
        os.makedirs(outdir, exist_ok=True)
        outfile = os.path.join(outdir, "{}_stat.png".format(self.sample))

        fig = plt.figure(figsize=(4, 7))

        ax1 = fig.add_subplot(411)
        ax1.imshow(contour_img)
        plt.title("tissue region")
        ax1.axis('off')
        ax1.axis('tight')

        ax3 = fig.add_subplot(412)

        if type(self.img) != np.ndarray:
            self.img = self.img.get()
        ax3.imshow(self.img)
        plt.title("sampling region")
        ax3.axis('off')
        ax3.axis('tight')

        ax4 = fig.add_subplot(413)
        ax4.hist(self.areas)
        ax4.set_xlabel('stained area(%)')

        ax5 = fig.add_subplot(414)
        ax5.hist(self.blurs)
        ax5.set_xlabel('blur level')

        fig.savefig(outfile, dpi=600)

    def write_log(self, outfile):
        fout = open(outfile, "w")
        print(f"datetime: {self.datetime}", file=fout)
        print(f"sample: {self.sample}", file=fout)
        print(f"Otsu threshold : {self.otsu_th}", file=fout)
        fout.close()

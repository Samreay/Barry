import logging
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import time
import numpy as np


class Viewer(object):
    def __init__(self, extents, parameters=None, truth=None, batch=100, tbatch=10):
        self.logger = logging.getLogger(__name__)
        self.initialised = False
        self.parameters = parameters
        self.extents = extents
        self.truth = truth
        self.figure = None
        self.axes = None
        self.dim = None
        self.backgrounds = None
        self.points = None
        self.batch = batch
        self.batched = []
        self.colours = cm.rainbow(np.linspace(0, 1, 20))
        self.index = 0
        self.last_time = None
        self.tbatch = tbatch
        self.pause = 0.1

    def initialise(self, position):
        self.logger.info("Initialising viewer")
        print(self.parameters)
        plt.ion()

        if self.parameters is not None:
            self.dim = len(self.parameters)
        else:
            self.dim = len(position)

        n = self.dim - 1
        self.pause = n * n * 0.5 * 0.1
        self.figure, self.axes = plt.subplots(n, n, figsize=(10, 10), squeeze=False)

        for i in range(n):
            for j in range(n):
                ax = self.axes[i, j]
                p1 = i
                p2 = j + 1
                param1 = None if self.parameters is None else self.parameters[p1]
                param2 = None if self.parameters is None else self.parameters[p2]
                print(i, j, param1, param2, self.extents[p1], self.extents[p2], position[p1], position[p2])
                display_x_ticks = False
                display_y_ticks = False
                if i < j:
                    ax.set_frame_on(False)
                    ax.set_xticks([])
                    ax.set_yticks([])
                else:
                    if i != n - 1:
                        ax.set_xticks([])
                    else:
                        display_x_ticks = True
                        ax.set_xlabel(param2, fontsize=14)
                    if j != 0:
                        ax.set_yticks([])
                    else:
                        display_y_ticks = True
                        ax.set_ylabel(param1, fontsize=14)
                    if display_x_ticks:
                        [l.set_rotation(45) for l in ax.get_xticklabels()]
                        ax.xaxis.set_major_locator(MaxNLocator(5, prune="lower"))
                    if display_y_ticks:
                        [l.set_rotation(45) for l in ax.get_yticklabels()]
                        ax.yaxis.set_major_locator(MaxNLocator(5, prune="lower"))
                    ax.set_ylim(self.extents[p1])
                    ax.set_xlim(self.extents[p2])
                    if self.truth is not None:
                        ax.axhline(self.truth[p1], dashes=(3, 3), ls="--", color="k")
                        ax.axvline(self.truth[p2], dashes=(3, 3), ls="--", color="k")

        self.points = [[self.axes[i, j].plot(position[j + 1], position[i], '.', alpha=0.7)[0]
                        for j in range(n) if i >= j] for i in range(n)]
        self.figure.show(False)
        self.figure.canvas.draw()
        plt.pause(self.pause)
        self.backgrounds = [[self.figure.canvas.copy_from_bbox(self.axes[i, j].bbox)
                           for j in range(n) if i >= j] for i in range(n)]
        self.initialised = True
        self.last_time = time.time()

    def update(self):
        self.logger.info("Updating viewer")
        data = np.vstack(tuple(self.batched))
        n = self.dim - 1
        for i in range(n):
            for j in range(n):
                if i < j:
                    continue
                self.axes[i][j].plot(data[:, j + 1], data[:, i], '.-', alpha=0.7, color=self.colours[self.index])
        self.index = (self.index + 1) % len(self.colours)
        self.figure.canvas.draw()
        plt.pause(self.pause)
        self.batched = []
        self.last_time = time.time()

    def callback(self, log_posterior, position, weight=1):
        if not self.initialised:
            self.initialise(position)
        else:
            self.batched.append(position)
            if len(self.batched) >= self.batch or time.time() - self.last_time > self.tbatch:
                self.update()

import argparse
import sys
#import numpy as np
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import asyncio
# local
import oxy
import mqtt
from mqtt import run_subscription, logger


class FigureData():

    def __init__(self, fig, data=None):
        self.fig = fig
        self.axs = fig.axes
        self.oo = self.get_oo_init(data)
        self.show()

    def show(self):
        self.oo.draw_sO2(self.axs)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def get_oo_init(self, oo_prev=None):
        oo = oxy.Oxy()
        if oo_prev is not None:
            oo.data = oo_prev.data.median().to_frame().T
        return oo

def get_options(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', dest='width', type=int, default=10)
    parser.add_argument('--height', dest='height', type=int, default=12)
    parser.add_argument('--dpi', dest='dpi', type=int, default=80)
    parser.add_argument('--debug', '-D', dest='debug', action='store_true')
    options = parser.parse_args()
    return options


if __name__ == '__main__':

    options = get_options(sys.argv[1:])

    if not options.debug:
        logger.propagate = False

    ## loop count
    #n = 10
    ## ... infinite
    n = None

    h = options.height
    w = options.width
    dpi = options.dpi

    ## graphics
    #plt.show()
    axs_n = 4
    fig, axs = plt.subplots(axs_n, 1,
                            sharex=True,
                            figsize=(w, h),
                            dpi=dpi,
                            tight_layout=True,
                            num='demo')
    fig.suptitle('From light intensities to Hb oxygen saturation.')

    show = FigureData(fig=fig)

    ## definition of mqtt consumer_action
    def consumer_action(buffer,
                        keep=show.oo, # keep will be merged with new oo data
                        show=show):
        # take new data
        oo = oxy.Oxy(buffer, do_eval=False)
        # update evaluation
        keep.eval()
        # update plot from keep
        show.show()
        # data will be joined in mqtt consumer function
        return oo

    ## asyncio loop
    if True:
    #if False:

        loop = asyncio.get_event_loop()
        _ = asyncio.ensure_future(run_subscription(consumer_action=consumer_action, 
                                                   produce_n=n, 
                                                   consume_n=n,
                                                   consume_keep=show.oo))

        try:
            loop.run_forever()
        finally:
            loop.close()

# %%
import matplotlib.pyplot as plt
#
import oxy

# %%
#do_save = False
do_save = True

fig, axs = plt.subplots(2, 1, sharex=True, num='Hb & HbO2 Spektren (und LEDs)')
oxy.draw_spectra(axs, wl_min=550, use_NIRb=True, use_beta=True, use_DE=True)
if do_save:
    fig.savefig('figs/basics/Hb_and_LEDs_Spektren-2.png', dpi=150)

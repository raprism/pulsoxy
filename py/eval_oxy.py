# coding: utf-8
import sys
import logging
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
#
from data_proc.signal import mfreqz, impz
import oxy

if True:
    #if False:
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
else:
    logging.basicConfig(stream=open('log.out', 'w'), level=logging.DEBUG)
logger = logging.getLogger(__file__)

do_save = False
#do_save = True

fig, axs = plt.subplots(2, 1, sharex=True, num='Hb & HbO2 spectra')
oxy.draw_spectra(axs)
if do_save:
    fig.savefig('figs/basics/Hb_and_LEDs_spectra.png', dpi=150)

## measurement example and its evaluation ##

if len(sys.argv) > 1:
    fn = sys.argv[1]
else:
    #fn = 'data/demo.csv'
    fn = 'data/demo2.csv'

fn_path = Path(fn)

tname = '.'.join(fn_path.name.split('.')[:-1])
ftmpl = tname + ' - #{}'

#oo = oxy.Oxy(fn)
oo = oxy.Oxy(fn, do_eval=False)
df = oo.data

if True:  # cut data
#if False:
    #df = df[df.time_sec <= 10]
    use_min_len = False
    use_end = False
    t_offs = 0
    #t_offs = 1
    if use_min_len:
        min_sample = 300 # minimum sample number
        t_len = int(np.ceil(min_sample * oo.meth.dt_avg))
    else:
        t_len = 60 # second
    if use_end:
        t_beg, t_end = df.time_sec.iloc[-1] + np.r_[-t_len-t_offs, -t_offs]
    else:
        t_beg, t_end = df.time_sec.iloc[0] + np.r_[t_offs, t_len]
    df = df.query('@t_beg <= time_sec <= @t_end')
    df.reset_index(inplace=True)
    #
    oo.update(df)
else:
    oo.eval()

# frequency det. in bpm
t = df.time_sec
dt_avg = t.diff().mean()
t_avg = np.arange(0, len(df) * dt_avg, dt_avg)[:len(df)]
t60_avg = t_avg / 60
t60 = t / 60
#Fmax = min(round(1. / t.diff().max(), 0), 200)
Fs = 1. / dt_avg
#Fny = 2 * pi * Fs/2 # in radians?
Fny = Fs / 2  # in radians?
Fshow = min(Fny * 60, 200)
freqs60 = np.r_[.1:Fshow + .1:.1].astype(float)

n = 4
ia = 0
fig, axs = plt.subplots(n, 1, sharex=True, num=ftmpl.format(1), figsize=(10, 10))
fig.suptitle('From light intensities to Hb oxygen saturation.')
step_text = oo.draw_sO2(axs)
if do_save:
    fig.savefig('figs/eval/eval_oxy-{}.png'.format(step_text), dpi=150)

## heart rate determination
pi = np.pi
ir_ac = df['IR_ac'].values
wave = ir_ac - np.median(ir_ac)
psd = scipy.signal.lombscargle(t60.values, wave, 2 * pi * freqs60)
psd_t_avg = scipy.signal.lombscargle(t60_avg, wave, 2 * pi * freqs60)

fig, axs = plt.subplots(num=ftmpl.format(2))
ax = axs
fig.suptitle("heart rate spectrum (from filtered 'AC part')")
ax.plot(freqs60, psd / psd.max())  #, label='')
#ax.plot(freqs, psd_t_avg, label='t_avg')
ax.set_xlabel('frequency (bpm)')
ax.set_ylabel('PSD (normed max. ampl.)')
ax.set_yticks([0, 1])

## linear filter characteristics
for i, (key, taps) in enumerate(oo.meth.taps.items()):
    fig = mfreqz(taps, fig_num=ftmpl.format(11 + 10*i))
    fig.suptitle('"{}"'.format(key))
    if do_save:
        fig.savefig('figs/basics/freqz_{}.png'.format(key), dpi=150)
    fig = impz(taps, fig_num=ftmpl.format(12 + 10*i))
    fig.suptitle('"{}"'.format(key))
    if do_save and False:
        fig.savefig('figs/basics/impz_{}.png'.format(key), dpi=150)

## sO2 step by step:
for n in range(3):
    size = 6 + n + 1
    fig, axs = plt.subplots(n+1, 1, sharex=True, figsize=(size, size))
    if not n:
        axs = [axs]
    step_text = oo.draw_sO2(axs)
    if do_save:
        fig.savefig('figs/eval/eval_oxy-{}.png'.format(step_text), dpi=150)

plt.show()

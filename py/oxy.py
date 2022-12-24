import io
import os
from collections import OrderedDict
from pathlib import Path
#
import numpy as np
import pandas as pd
import scipy.signal
#
from data_proc.signal import get_peaks

## hb_wl_points.to_json()
## http://omlc.org/spectra/hemoglobin/summary.html (by Scott Prahl)
json_hb_wl_points_1 = """
{"wl":  {"RED":660,    "IR":880},
 "Hb02":{"RED":319.6,  "IR":1154.0},
  "Hb": {"RED":3226.56,"IR":726.44}}"""
## Bosschaart et al 2013 (https://www.researchgate.net/publication/257754996_A_literature_review_and_novel_theoretical_approach_on_the_optical_properties_of_whole_blood)
json_hb_wl_points_2 = """
{"wl":  {"RED":660,    "IR":880},
 "Hb02":{"RED":0.15,  "IR":0.56},
  "Hb": {"RED":1.64,"IR":0.44}}"""
#json_hb_wl_points = json_hb_wl_points_1
json_hb_wl_points = json_hb_wl_points_2

## settings
res_colors = OrderedDict(zip(['sO2', 'R'], ['#14AFF0ee', '#ff5500ff']))
leds_colors = OrderedDict(zip(['IR', 'RED'], ['#A38C89', '#C82C1C']))
leds = list(leds_colors.keys())
lcolors = list(leds_colors.values())
#
wl_RED, fwhm_RED = 660, 20
wl_NIRa, fwhm_NIRa = 880, 30
wl_NIRb, fwhm_NIRb = 950, 30


def get_I(wl_c, FWHM, norm_ampl=True):
    """
    gaussian curve as function of
    wl_c - center wavelength
    FWHM - full width (at) half maximum
    """
    wl_lim = int(2 * FWHM)
    wl = np.r_[int(wl_c - wl_lim):int(wl_c + wl_lim + 1)]
    sigma = FWHM / 2.35
    I = scipy.stats.norm.pdf(wl, wl_c, sigma)
    if norm_ampl:
        I /= I.max()
    return wl, I


def get_ac_dc(vec, taps_ac, taps_dc):
    """
    separate AC and DC part of vec
    by means of FIR filters as defined by
    taps_ac and taps_dc
    """
    try:
        vec = np.asarray(vec)
        ## DC baseline
        ## variant: use moving median filter
        #dc = scipy.signal.medfilt(vec, 101)
        dc = scipy.signal.filtfilt(taps_dc, [1.], vec)
        ## variant: hpf for direct AC pre-calculation
        #_ac = scipy.signal.filtfilt(b_hpf, a_hpf, vec)
        ## rest belongs to AC part, inverted as absorption-like value
        _ac = -(vec - dc)
        ## lpf
        ac = scipy.signal.filtfilt(taps_ac, [1.], _ac)
        ## minimum of AC part has actually still an offset above zero
        ## shift it just above zero for easier post-processing
        if False:
            #if True:
            if ac.min() > 0:
                ac -= ac.min() * (1 - 1E-6)
            else:
                ac += ac.min() * (1 + 1E-6)
        ## correct DC accordingly
        #dc += 0.5 * (ac.max() - ac.min())
        if False:
            import logging
            logging.info(vec)
    except ValueError:
        if False:
            print(len(vec))
            from IPython.core.debugger import set_trace
            set_trace()
        else:
            import logging
            logging.info(len(vec))
        nan_vec = np.nan * vec
        ac, dc = nan_vec, nan_vec
    return ac, dc


def get_sO2(r, eps):
    return ( eps.loc['RED'].Hb - eps.loc['IR'].Hb * r ) \
        / ( eps.loc['RED'].Hb - eps.loc['RED'].Hb02 \
            + ( eps.loc['IR'].Hb02 - eps.loc['IR'].Hb ) * r )


def get_hb_spectra(prefix='.'):
    fn = Path(prefix) / 'data' / 'Hb_ext.csv'
    hb_ext = pd.read_csv(fn, sep=';', comment='#')
    return hb_ext.rename(columns={'lambda': 'wl'})


## plotting
def draw_spectra(axs,
                 wl_min=500,
                 use_NIRb=False,
                 use_beta=False,
                 use_DE=False):
    # # compare https://omlc.org/spectra/hemoglobin/
    # # and Pulsoximeter script ()
    assert len(axs) > 1
    ## nominal spectra and LED intensities ##
    hb_ext = get_hb_spectra()
    wl_range_RED, I_RED = get_I(wl_RED, fwhm_RED)
    if use_NIRb:
        wl_NIR = wl_NIRb
        wl_range_NIR, I_NIR = get_I(wl_NIRb, fwhm_NIRb)
    else:
        wl_NIR = wl_NIRa
        wl_range_NIR, I_NIR = get_I(wl_NIRa, fwhm_NIRa)
    ## spectra points at wls
    if use_beta:
        # # M = mol/l -> Mol / (cm^3 * 1000)
        # # and per Fe: factor 1000 / 4 = 250
        hb_ext[['Hb02', 'Hb']] *= 250E-6
        ylabel_ext = r'$\beta$ (in $10^6$ cm$^{2}$/Mol)'
    else:
        ylabel_ext = r'$\epsilon$ (in cm$^{-1}$M$^{-1}$)'
    hb_wl_points = hb_ext.set_index('wl').loc[[wl_RED, wl_NIR]]\
                                         .reset_index()
    hb_wl_points.index = ['RED', 'IR']
    ia = 0
    hb_ext[hb_ext.wl >= wl_min].plot(x='wl',
                                     ax=axs[ia],
                                     color=['g', 'b'],
                                     logy=True)
    ymin, ymax = axs[ia].get_ylim()
    for cidx, row in hb_wl_points.iterrows():
        color = leds_colors[cidx]
        axs[ia].vlines(row[0], ymin, ymax, color, 'dotted')
        axs[ia].plot([row[0]] * 2, row[1:], 'o', color=color)
    axs[ia].set_ylabel(ylabel_ext)
    # # LEDs
    ia += 1
    if use_DE:
        ylabel_LED = 'LED Intensität \n(norm. Ampl.)'
        plabel_RED = 'Rot'
    else:
        ylabel_LED = 'LED intensity \n(normed ampl.)'
        plabel_RED = 'red'
    axs[ia].plot(wl_range_RED,
                 I_RED,
                 color=leds_colors['RED'],
                 label=plabel_RED)
    axs[ia].plot(wl_range_NIR, I_NIR, color=leds_colors['IR'], label='NIR')
    axs[ia].legend()
    axs[ia].set_yticks([0, 1])
    axs[ia].set_ylabel(ylabel_LED)
    axs[ia].set_xlabel(r'$\lambda$ (in nm)')


## classes


class Meth:

    def __init__(self, oxy_data, min_len=300):
        ## assume that taps can be calculated only once
        ## min_len actually needs to be taken from filter characteristics
        self.data = None
        self.taps = {}
        self.dt_avg = None
        self.min_len = min_len
        self._new_data(oxy_data)
        self.hb_wl_points = pd.read_json(json_hb_wl_points)

    def _new_data(self, data):
        self.data = data
        self.results = {}
        if len(self.data):
            self._prep_data()
            if not self.taps:
                self.taps = self._set_taps()

    def update(self, data):
        if id(data) != id(self.data):
            self._new_data(data)

    def _prep_data(self, time0=False):
        if time0:
            self.data.loc[:, 'time_sec'] = 1E-3 * (self.data.time_ms -
                                                   self.data.time_ms.iloc[0])
        else:
            self.data.loc[:, 'time_sec'] = 1E-3 * self.data.time_ms
        self.dt_avg = self.data.time_sec.diff().mean()

    def _set_taps(self):
        if self.data is not None \
           and len(self.data) >= self.min_len:
            try:
                Fs = 1. / self.dt_avg
                # Fny = Fs / 2
                window = ('kaiser', 3.)
                ## filter for AC part
                taps_lpf = scipy.signal.firwin(51, 3, fs=Fs, window=window)
                ## filter for DC part
                taps_dc = scipy.signal.firwin(51, 0.3, fs=Fs, window=window)
                return dict(lpf=taps_lpf, dc=taps_dc)
            except ValueError:
                return {}
        else:
            return {}

    def _set_ac_dc(self, col, low_lim):
        ac, dc, _ac, _dc = [np.array(np.nan)] * 4
        if self.taps:
            _ac, _dc = get_ac_dc(self.data[col], self.taps['lpf'],
                                 self.taps['dc'])
        if (abs(_ac) >= low_lim).any():
            ac, dc = _ac, _dc
        try:
            self.data.loc[:, col + '_ac'] = ac
            self.data.loc[:, col + '_dc'] = dc
        except ValueError:
            breakpoint()

    def set_ac_dc(self, low_lim=None):
        for col in ['RED', 'IR']:
            self._set_ac_dc(col, low_lim)

    def _get_ac_dc(self):
        return self.data[['RED_ac', 'RED_dc', 'IR_ac', 'IR_dc']].values.T

    @staticmethod
    def _get_r_oxysat(red_ac, red_dc, ir_ac, ir_dc):
        if True:
            _r_oxysat = np.log((red_ac + red_dc) / red_dc) \
                        / np.log((ir_ac + ir_dc) / ir_dc)
        else:
            ## variant uses ln((ac+dc)/dc) = ln(1 + ac/dc) ~ ac/dc
            _r_oxysat = (red_ac / red_dc) / (ir_ac / ir_dc)
        return _r_oxysat

    def get_r_oxysat(self):
        df = self.data
        mpd = int(0.5 / self.dt_avg)
        # take at peaks
        IR_ac_high = get_peaks(df, 'IR_ac', mph=df.IR_ac.max() * 0.5,
                               mpd=mpd).rename('IR_ac_high')
        IR_ac_low = get_peaks(df,
                              'IR_ac',
                              mph=df.IR_ac.max() * 0.5,
                              mpd=mpd,
                              valley=True).rename('IR_ac_low')
        RED_ac_high = get_peaks(df,
                                'RED_ac',
                                mph=df.RED_ac.max() * 0.5,
                                mpd=mpd).rename('RED_ac_high')
        RED_ac_low = get_peaks(df,
                               'RED_ac',
                               mph=df.RED_ac.max() * 0.5,
                               mpd=mpd,
                               valley=True).rename('RED_ac_low')
        ac = pd.concat([IR_ac_high, IR_ac_low, RED_ac_high, RED_ac_low],
                       axis=1)
        ## this gives max. 2 peak and 2 valley per heart beat
        #ac = ac.rolling(window=41, min_periods=1)\
        #      .median().abs().dropna()
        ac = ac.rolling(window=21, min_periods=1,
                        #win_type='hanning').mean().abs().dropna()
                        win_type='kaiser')\
               .mean(beta=3.).abs().dropna()
        ix_r = ac.index
        r_oxysat = self._get_r_oxysat(ac.RED_ac_high + ac.RED_ac_low,
                                      df.RED_dc.loc[ix_r],
                                      ac.IR_ac_high + ac.IR_ac_low,
                                      df.IR_dc.loc[ix_r])
        return r_oxysat, ix_r

    def get_sO2(self, percentage=True):
        r_oxysat, ix_r = self.get_r_oxysat()
        sat_OS = pd.concat([self.data.time_sec[ix_r],
                            r_oxysat.rename('R')],
                           axis=1)
        sat_OS.loc[:, 'sO2'] = get_sO2(sat_OS.R, self.hb_wl_points)
        if percentage:
            sat_OS.sO2 *= 100.0
        results = dict(sat_OS=sat_OS,
                       sO2_mean=sat_OS.sO2.mean(),
                       r_OS_mean=sat_OS.R.mean())
        return results

    def eval(self, low_lim=None, verbose=True):
        self.set_ac_dc(low_lim=low_lim)
        if len(self.data) < self.min_len:
            return {}
        ## SaO2 and r factor of oxy saturation (with peak times)
        self.results = results = self.get_sO2()
        if verbose:
            ## for a check use simply ac max and dc mean values
            red_ac, red_dc, ir_ac, ir_dc = self._get_ac_dc()
            r_check = ((red_ac.max() - red_ac.min()) / red_dc.mean()) / (
                (ir_ac.max() - ir_ac.min()) / ir_dc.mean())
            print("'quick' R calc gives {:.2f}".format(r_check))
            if np.isfinite(results['sO2_mean']):
                print('SaO2 = {}%'.format(int(round(results['sO2_mean']))))
        return results


class Oxy:

    cols = [
        'time_ms',
        'RED',
        'IR',
        #'HR',
        #'HRvalid',
        #'SPO2',
        #'SPO2valid',
    ]

    def __init__(self,
                 fn_or_bytes=bytearray(b'0;0;0'),
                 load=True,
                 do_eval=True):
        if isinstance(fn_or_bytes, str) \
        and os.path.isfile(fn_or_bytes):
            self.buffer = open(fn_or_bytes, 'r')
        elif isinstance(fn_or_bytes, bytearray):
            string = fn_or_bytes.decode('ascii')
            self.buffer = io.StringIO(string)
        else:
            raise NotImplementedError
        self.data = None
        self.meth = None
        self.results = {}
        if load:
            self.load(do_eval)

    def load(self, do_eval=True):
        if not self.buffer.closed:
            self.data = pd.read_csv(self.buffer,
                                    delimiter=';',
                                    header=None,
                                    names=self.cols)
            self.buffer.close()
        else:
            # empty df
            self.data = pd.DataFrame(columns=self.cols)
        self.meth = Meth(self.data)
        if do_eval:
            self.eval()

    def update(self, data, do_eval=True):
        self.data = data
        if do_eval:
            self.eval()

    def eval(self, low_lim=50, verbose=False):
        self.meth.update(self.data)
        self.results.update(self.meth.eval(low_lim=low_lim, verbose=verbose))

    def draw_sO2(self, axs):
        tmpl = 'step{}'
        if len(axs):
            ia = 0
            ax = axs[ia]
            ax.clear()
            #fig, axs = plt.subplots(2, 1, sharex=True)
            self.data.plot(x='time_sec',
                           y=leds,
                           ax=ax,
                           color=lcolors,
                           style='-')
            x = self.data.time_sec

            # # xlim sets uniformly x axis limits
            def xlim():
                axs[0].set_xlim(
                    np.floor(x.min()).round(0),
                    np.ceil(x.max() + 0.5).round(0))

            ax.set_ylabel('meas. intensity (a.u.)')
        if len(axs) > 1:
            ia += 1
            ax = axs[ia]
            ax.clear()
            self.data.plot(x='time_sec',
                           y=[led + '_dc' for led in leds],
                           ax=ax,
                           color=lcolors,
                           style='.-')
            ax.set_ylabel('determined DC part')
        else:
            xlim()
            return tmpl.format('_1')
        if len(axs) > 2:
            ia += 1
            ax = axs[ia]
            ax.clear()
            self.data.plot(x='time_sec',
                           y=[led + '_ac' for led in leds],
                           ax=ax,
                           color=lcolors,
                           style='.-')
            ax.set_ylabel('determined AC part')
        else:
            xlim()
            return tmpl.format('s_1to2')
        if len(axs) > 3:
            ia += 1
            ax = axs[ia]
            ax.clear()
            if hasattr(ax, 'right_ax'):
                ax.right_ax.clear()
            ax.set_ylabel(r'$R_{OS}$')
            ax.set_xlabel('meas. time (in seconds)')
            res = self.results
            if res and 'sat_OS' in res and len(res['sat_OS']):
                # use of secondary_y option is buggy!
                sat_OS = res['sat_OS'].set_index('time_sec')
                label = [
                    r'avg. $R_{{OS}}$ of {:.2f}'.format(res['r_OS_mean']),
                    r'avg. $S^{{unc}}_{{O2}}$% of {:.0f}%'.format(
                        res['sO2_mean'])
                ]
                sat_OS['R'].plot(
                    #x='time_sec',
                    y='R',
                    ax=ax,
                    linewidth=0,
                    marker='o',
                    color=res_colors['R'],
                    #label=label,
                )
                sat_OS['sO2'].plot(
                    #x='time_sec',
                    secondary_y='sO2',
                    ax=ax,
                    linewidth=0,
                    marker='o',
                    alpha=0.3,
                    color=res_colors['sO2'],
                    #label=label,
                )
                ax.right_ax.set_ylabel(r'$S^{unc}_{O2}$% (uncalibrated)')
                ax.right_ax.legend(label[1:],
                                   loc=1,
                                   markerscale=0.35,
                                   fontsize=14,
                                   framealpha=.5)
                ax.legend(label[:1], loc=2, markerscale=0.35,
                          fontsize=16)  # framealpha=.5
        else:
            xlim()
            return tmpl.format('s_1to3')
        if len(axs):
            xlim()
            return tmpl.format('s_all')
        else:
            return 'NO_DATA'

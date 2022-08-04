# External imports
import medusa.spectral_analysis
import numpy as np
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import scipy.signal as ss
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, \
    SGDClassifier
# Medusa imports
from medusa import components
from medusa import epoching
from medusa.frequency_filtering import IIRFilter
from medusa.spectral_analysis import normalize_psd
from medusa.plots import topographic_plots


class EEGBlinkDetection:
    def __init__(self):
        # Initialize attributes
        self.recording_signal = None
        self.time = None
        self.channel_set = None
        self.fs = None
        self.event_data = None
        self.blink_channels = None
        self.hp_filter = None
        self.lp_filter = None
        self.blink_signal = None
        self.blink_signal_mask = None
        self.show_blink_signal = None
        self.output_signal = None
        self.rest_epochs = None
        self.rest_block_duration = None
        self.blink_epochs = None
        self.blink_block_duration = None
        self.algorithm = None

    def load_recording(self, path):
        try:
            # Extract signal and attributes
            recording = components.Recording.load_from_bson(path)
            self.recording_signal = recording.eeg.signal
            self.time = recording.eeg.times
            self.channel_set = recording.eeg.channel_set
            self.fs = recording.eeg.fs
            self.event_data = recording.eegblinkdata
            self.blink_block_duration = np.min(
                self.event_data.blink_block_end - self.event_data.blink_block_init)
            self.rest_block_duration = np.min(
                self.event_data.rest_block_end - self.event_data.rest_block_init)

            # Apply a 0.5 Hz zero-phase high-pass filter to the EEG
            self.hp_filter = IIRFilter(order=3, cutoff=[0.5,40], btype='bandpass')
            self.hp_filter.fit(fs=self.fs, n_cha=len(self.channel_set.l_cha))
            self.output_signal = self.hp_filter.transform(self.recording_signal)

            # Extract epochs from blink blocks before applying low pass filter
            self.show_blink_signal = epoching.get_epochs_of_events(
                timestamps=self.time,
                signal=self.output_signal,
                onsets=self.event_data.blink_block_init,
                fs=self.fs,
                w_epoch_t=[0, int(self.blink_block_duration *
                                  1000)])
            self.show_signal(self.show_blink_signal)
        except Exception as e:
            print(e)

    @staticmethod
    def reshape_signal(epochs):
        try:
            epoch_c = epochs.copy()
            blocks, samples_per_block, channels = epoch_c.shape
            epoch_c = np.reshape(epoch_c,
                                 (int(blocks * samples_per_block), channels))
            return epoch_c
        except Exception as e:
            print(e)

    def show_signal(self, epoch, ch_labels=None):
        try:
            if ch_labels is None:
                ch_labels = self.channel_set.l_cha

            blocks, samples_per_block, channels = epoch.shape
            epoch_c = self.reshape_signal(epoch)

            channel_offset = np.zeros(epoch_c.shape[1])
            if len(channel_offset) > 1:
                channel_offset[1:] = np.max(np.max(epoch_c[:, 1:], axis=0) -
                                            np.min(epoch_c[:, :-1], axis=0))
                channel_offset = np.cumsum(channel_offset)
            epoch_c = epoch_c - channel_offset
            max_val, min_val = epoch_c.max(), epoch_c.min()
            display_times = np.linspace(0, int(epoch_c.shape[0] / self.fs),
                                        epoch_c.shape[0])

            # Matplotlib
            plt.plot(display_times, epoch_c, 'k', linewidth=0.5)
            plt.yticks(-channel_offset, labels=ch_labels)
            vertical_lines = np.empty((blocks - 1, 2, 50))
            for block in range(blocks - 1):
                vertical_lines[block, :, :] = np.asarray([np.ones(50) * (
                            block + 1) * int(samples_per_block / self.fs),
                                                          np.linspace(min_val,
                                                                      max_val,
                                                                      50)])
                plt.plot(vertical_lines[block, 0, :],
                         vertical_lines[block, 1, :], '--', color='red',
                         linewidth=1.5)
            plt.show(block=True)

        except Exception as e:
            print(e)

    def find_peaks(self, threshold, t_extend, signal):
        self.blink_signal_mask = np.zeros((signal.shape[0]))
        sorted_blink_idx = np.flip(np.argsort(signal))
        peak_idx = sorted_blink_idx[:int(signal.shape[0] * 0.1)]
        peak_sign = np.median(np.sign(signal[peak_idx]))

        # Detects blinks in the EEG for samples which are above the threshold
        self.blink_signal_mask = (signal * -peak_sign) > threshold
        # self.blink_signal_mask = (signal > threshold_up) | (-signal > threshold_down)
        if t_extend > 0:
            b = np.ones(int(t_extend * self.fs))
            self.blink_signal_mask = ss.filtfilt(b, 1, self.blink_signal_mask,
                                                 axis = 0)
            self.blink_signal_mask = self.blink_signal_mask > 0
            self.blink_signal_mask = np.squeeze(self.blink_signal_mask)
            # Clear sequences of ones in mask that are shorter than min_length
            peaks, pp = ss.find_peaks(self.blink_signal_mask,
                                      width=int(0.1 * self.fs),
                                      rel_height=1)
            self.blink_signal_mask = np.zeros(len(self.blink_signal_mask))
            for peak in range(len(peaks)):
                st_w = pp['left_bases'][peak]
                fn_w = pp['right_bases'][peak]
                self.blink_signal_mask[st_w:fn_w] = 1

    def preprocess_data(self, l_cha_blink_reference,threshold = 25,
                        t_extend=0.025):

        self.blink_channels = l_cha_blink_reference
        self.blink_channel_index = self.channel_set.get_cha_idx_from_labels(
            self.blink_channels)

        # Apply a low-pass filter to the chosen blink channels signals for blink detection
        self.lp_filter = IIRFilter(order=2, cutoff=5, btype='lowpass')
        self.lp_filter.fit(fs=self.fs, n_cha=len(self.blink_channels))
        self.blink_signal = self.lp_filter.transform(
            self.output_signal.copy()[:, self.blink_channel_index])

        # Events to epochs
        self.rest_epochs = epoching.get_epochs_of_events(timestamps=self.time,
                                                         signal=self.output_signal,
                                                         onsets=self.event_data.rest_block_init,
                                                         fs=self.fs,
                                                         w_epoch_t=[-1000,
                                                                    int(self.rest_block_duration *
                                                                        1000)])
        n_rest_epochs, len_rest_epochs = self.rest_epochs.shape[0], \
                                         self.rest_epochs.shape[1]

        self.blink_epochs = epoching.get_epochs_of_events(timestamps=self.time,
                                                          signal=self.blink_signal,
                                                          onsets=self.event_data.blink_block_init,
                                                          fs=self.fs,
                                                          w_epoch_t=[-1000,
                                                                     int(self.blink_block_duration *
                                                                         1000)])

        unfiltered_blink_epochs = epoching.get_epochs_of_events(
            timestamps=self.time,
            signal=self.output_signal,
            onsets=self.event_data.blink_block_init,
            fs=self.fs,
            w_epoch_t=[-1000, int(self.blink_block_duration *
                                  1000)])
        n_blink_epochs, len_blink_epochs = self.blink_epochs.shape[0], \
                                           self.blink_epochs.shape[1]

        # Show rest epochs for choosing bad epochs to remove those which have artifacts
        self.show_signal(self.rest_epochs)

        rest_epoch_to_remove_idx = input(
            'Select rest epochs to remove (Split by "," please):').split(',')

        if len(rest_epoch_to_remove_idx) >= 1 and rest_epoch_to_remove_idx[
            0] != '':
            for epoch_idx in range(len(rest_epoch_to_remove_idx)):
                rest_epoch_to_remove_idx[epoch_idx] = int(
                    rest_epoch_to_remove_idx[epoch_idx])
        else:
            rest_epoch_to_remove_idx = []

        if len(rest_epoch_to_remove_idx) > 0:
            self.show_signal(
                np.delete(self.rest_epochs, rest_epoch_to_remove_idx, axis=0))
            n_rest_epochs = n_rest_epochs - len(rest_epoch_to_remove_idx)
            print('Resting epochs without artifacts')

        # Reshape blink epochs
        self.blink_epochs_rs = np.squeeze(
            self.reshape_signal(self.blink_epochs))

        # Find the median sign of the 10% highest peaks
        self.find_peaks(threshold, t_extend, self.blink_epochs_rs)
        self.blink_signal_mask = self.blink_signal_mask.reshape(
            (n_blink_epochs, len_blink_epochs))
        cleaned_signal = None
        signal_labels = None
        for e in range(np.max([n_rest_epochs, n_blink_epochs])):
            if e >= n_rest_epochs:
                pass
            else:
                if e in rest_epoch_to_remove_idx:
                    pass
                else:
                    if cleaned_signal is None:
                        cleaned_signal = self.rest_epochs[e, :, :]
                        signal_labels = np.zeros(len_rest_epochs)
                    else:
                        cleaned_signal = np.vstack(
                            (cleaned_signal, self.rest_epochs[e, :, :]))
                        signal_labels = np.hstack(
                            (signal_labels, np.zeros(len_rest_epochs)))
            if e >= n_blink_epochs:
                pass
            else:
                if cleaned_signal is None:
                    cleaned_signal = unfiltered_blink_epochs[e, :, :]
                    cleaned_signal = np.vstack(
                        (cleaned_signal, unfiltered_blink_epochs[e, :, :]))
                    signal_labels = self.blink_signal_mask[e, :]
                    signal_labels = np.hstack(
                        (signal_labels, self.blink_signal_mask[e, :]))
                else:
                    cleaned_signal = np.vstack(
                        (cleaned_signal, unfiltered_blink_epochs[e, :, :]))
                    cleaned_signal = np.vstack(
                        (cleaned_signal, unfiltered_blink_epochs[e, :, :]))
                    signal_labels = np.hstack(
                        (signal_labels, self.blink_signal_mask[e, :]))
                    signal_labels = np.hstack(
                        (signal_labels, self.blink_signal_mask[e, :]))

        cleaned_signal = cleaned_signal - np.mean(cleaned_signal, axis=0)
        self.algorithm = SGEYESUB()
        # Equalize labels
        data_eq, labels_eq = self.algorithm.equalize_labels(cleaned_signal,
                                                            signal_labels)
        self.algorithm.fit(data=data_eq, labels=labels_eq)
        corrected = self.algorithm.apply(data=cleaned_signal, mask = None)

    def evaluation(self, path):
        from scipy.signal import  welch
        self.load_recording(path)
        test_signal = self.output_signal - np.mean(self.output_signal, axis=0)
        corrected = self.algorithm.apply(data=test_signal,mask = None)
        test_rest_epochs = epoching.get_epochs_of_events(timestamps=self.time,
                                                         signal=corrected,
                                                         onsets=self.event_data.rest_block_init,
                                                         fs=self.fs,
                                                         w_epoch_t=[-1000,
                                                                    int(self.rest_block_duration *
                                                                        1000)])
        uncorrected_rest_epochs = epoching.get_epochs_of_events(timestamps=self.time,
                                                         signal=test_signal,
                                                         onsets=self.event_data.rest_block_init,
                                                         fs=self.fs,
                                                         w_epoch_t=[-1000,
                                                                    int(self.rest_block_duration *
                                                                        1000)])
        _, psd_corrected = welch(test_rest_epochs, fs = self.fs, window='hamming',axis=1,
                                 nperseg=768,scaling='spectrum',
                                 noverlap = int(768*0.9))
        _, psd_uncorrected = welch(uncorrected_rest_epochs, fs=self.fs, window='hamming',
                                 axis=1,nperseg=768,scaling='spectrum',
                                 noverlap = int(768*0.9))

        psd_corrected= medusa.spectral_analysis.normalize_psd(psd_corrected)
        psd_uncorrected= medusa.spectral_analysis.normalize_psd(psd_uncorrected)
        psd_corrected = np.mean(psd_corrected,axis = 0)
        psd_uncorrected = np.mean(psd_uncorrected, axis = 0)

        psd_ratio = psd_corrected/psd_uncorrected
        frontales = psd_ratio[:, :6]
        frontales_m = np.mean(frontales, axis=1)
        plt.plot(_, 10*np.log(frontales_m))
        centrales = psd_ratio[:, 6:11]
        centrales_m = np.mean(centrales, axis=1)
        plt.plot(_, 10*np.log(centrales_m))
        parietales = psd_ratio[:, 11:]
        parietales_m = np.mean(parietales, axis=1)
        plt.plot(_, 10*np.log(parietales_m))
        # plt.ylim(0.5, 1.2)
        plt.xlim(0.5, 40)

        # RMSE
        n_epochs = uncorrected_rest_epochs.shape[0]
        rmse = np.zeros((n_epochs, 1, 16))
        for n in range(n_epochs):
            for c in range(16):
                rmse[n, :, c] = np.sqrt(
                    mean_squared_error(uncorrected_rest_epochs[n, :, c],
                                       test_rest_epochs[n, :, c], ))

        rmse = np.mean(rmse, axis=0)
        topographic_plots.plot_topography(self.channel_set, values=rmse,cmap='plasma',
                                          clim=[0,10],plot_extra=False)


class SGEYESUB:
    def __init__(self):
        # Properties
        self.W = None  # Unmixing Matrix
        self.A = None  # Mixing Matrix
        self.C = None  # Correction Matrix
        self.plr_lambra_l2 = 1
        self.plr_lambda_l1 = 0.01
        self.plr_tol = 0.001
        self.plt_maxiter = 10000

    def fit(self, data, labels, ):
        # self.W = LogisticRegression(penalty='elasticnet', tol=self.plr_tol,
        #                             class_weight='balanced', solver='saga',
        #                             max_iter=self.plt_maxiter, C=0.1,
        #                             l1_ratio=0.01).fit(data, labels)
        # self.W = LogisticRegressionCV(penalty='elasticnet', tol=self.plr_tol,
        #                             class_weight='balanced', solver='saga',
        #                             max_iter=self.plt_maxiter, l1_ratios=[0.1,0.01,0.001],
        #                               cv=4).fit(data, labels)
        self.W = SGDClassifier(loss="hinge",penalty='elasticnet', tol=self.plr_tol,
                                    class_weight='balanced',
                                    max_iter=self.plt_maxiter, l1_ratio=0.01,
                                    early_stopping=True).fit(data, labels)
        self.W = self.W.coef_.T
        data = data.T
        print('Weights fitted')
        R = np.cov(data)
        R_subspace = np.matmul(np.matmul(self.W.T, R), self.W)
        self.A = np.matmul(np.matmul(R, self.W), np.linalg.inv(R_subspace))
        self.C = np.eye(data.shape[0]) - np.matmul(self.A, self.W.T)
        self.C = self.C.T

    def apply(self, data,mask = None):
        if mask is not None:
            mask_no_blinks = np.ones((mask.shape))
            mask_no_blinks[np.where(mask == 1)[0]] = 0

            # lpf = IIRFilter(order=1,cutoff=25,btype="lowpass")
            # mask = lpf.fit_transform(mask[:,np.newaxis],256)

            data_no_blinks = mask_no_blinks[:,np.newaxis] * data
            data_blinks = mask[:,np.newaxis] * np.matmul(data,self.C)
            corrected = data_no_blinks + data_blinks
        else:
            corrected = np.matmul(data, self.C)
        return corrected

    def equalize_labels(self, data, labels):
        n_l = np.zeros(2)
        for l in range(len(n_l)):
            n_l[l] = np.sum(labels == l)
        n_max = n_l.min()
        labels_out = None
        data_out = None
        for l in range(len(n_l)):
            l_idx = np.where(labels == l)[0]
            idxs = np.arange(n_max) % n_l[l]
            l_idx = l_idx[idxs.astype(int)]
            if data_out is None:
                data_out = data[l_idx, :]
                labels_out = labels[l_idx]
            else:
                data_out = np.vstack((data_out, data[l_idx, :]))
                labels_out = np.hstack((labels_out, labels[l_idx]))

        return data_out, labels_out


class EEGBlinkData(components.ExperimentData):
    """Experiment info class for Blink correction. It records
       both resting and blink epochs to extract the blink subspace and
       extract it from EEG."""

    def __init__(self, rest_block_presentation, rest_block_init, rest_block_end,
                 blink_block_presentation, blink_block_init, blink_block_end,
                 blik_event):

        self.rest_block_presentation = rest_block_presentation
        self.rest_block_init = rest_block_init
        self.rest_block_end = rest_block_end
        self.blink_block_presentation = blink_block_presentation
        self.blink_block_init = blink_block_init
        self.blink_block_end = blink_block_end
        self.blik_event = blik_event

    def to_serializable_obj(self):
        rec_dict = self.__dict__
        for key in rec_dict.keys():
            if type(rec_dict[key]) == np.ndarray:
                rec_dict[key] = rec_dict[key].tolist()
        return rec_dict

    @classmethod
    def from_serializable_obj(cls, dict_data):
        rest_block_presentation = np.asarray(
            dict_data['rest_block_presentation'])
        rest_block_init = np.asarray(dict_data['rest_block_init'])
        rest_block_end = np.asarray(dict_data['rest_block_end'])
        blink_block_presentation = np.asarray(
            dict_data['blink_block_presentation'])
        blink_block_init = np.asarray(dict_data['blink_block_init'])
        blink_block_end = np.asarray(dict_data['blink_block_end'])
        blik_event = np.asarray(dict_data['blik_event'])
        return cls(rest_block_presentation=rest_block_presentation,
                   rest_block_init=rest_block_init,
                   rest_block_end=rest_block_end,
                   blink_block_presentation=blink_block_presentation,
                   blink_block_init=blink_block_init,
                   blink_block_end=blink_block_end,
                   blik_event=blik_event)


if __name__ == '__main__':
    from medusa import meeg
    from medusa import components
    from medusa.artifact_rejection import EEGBlinkDetection

    path = 'D:\Scripts Estudios\Corrección de Artefactos Online\Registros/ana_3.rec.bson'
    blink = EEGBlinkDetection()
    blink.load_recording(path)
    blink.preprocess_data(['F8'])
    blink.evaluation('D:\Scripts Estudios\Corrección de Artefactos Online\Registros/diego_2.rec.bson')
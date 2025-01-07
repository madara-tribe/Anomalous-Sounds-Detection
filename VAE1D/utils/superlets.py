import numpy as np
import mne


def apply_superlet(audio_data, sampling_rate):
    t = np.linspace(0, len(audio_data) / sampling_rate, len(audio_data))

    # Superlets Transform
    sfreq = sampling_rate  # Sampling frequency
    freqs = np.linspace(1, 50, 100)  # Frequencies from 1Hz to 50Hz

    # Using MNE's tfr_array_morlet to calculate superlet
    power_superlets = mne.time_frequency.tfr_array_morlet(
        audio_data[np.newaxis, np.newaxis, :],  # Shape (n_epochs, n_channels, n_times)
        sfreq=sfreq,
        freqs=freqs,
        n_cycles=2,  # Number of cycles for Morlet wavelet
        use_fft=True,
        output='power',
        decim=1,
        n_jobs=1
    )
    return power_superlets
    #print(power_superlets.shape)
    #signal = power_superlets.flatten()
    #fs = sampling_rate
    # Plotting Superlets Power
    #fig, ax = plt.subplots(figsize=(15, 2), dpi=300)
    #ax.set_xlabel("Time (ms)")
    #ax.plot(jnp.linspace(0, len(signal)/fs, len(signal)), signal)

    #plt.figure(figsize=(12, 6))
    #plt.imshow(
    #    power_superlets[0, 0], aspect='auto', origin='lower',
    #    extent=[t[0], t[-1], freqs[0], freqs[-1]]
    #)
    #plt.colorbar(label='Power')
    #plt.xlabel('Time (s)')
    #plt.ylabel('Frequency (Hz)')
    #plt.title('Superlets Transform Power')
    #plt.show()

import streamlit as st
import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
import librosa.display
import os
import tempfile
import soundfile as sf
import matplotlib.pyplot as plt
import time

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

st.title("Speech Command Recognition DemoApp")

def load_commands():
    return ['left', 'dog', 'three', 'five', 'up', 'cat', 'zero', 'happy', 'tree', 'wow',
            'right', 'six', 'yes', 'four', 'nine', 'on', 'sheila', 'eight', 'down', 'seven',
            'no', 'bird', 'stop', 'go', 'off', 'house', 'two', 'marvin', 'one', 'bed']

def show_waveform(audio, sample_rate):
    fig, ax = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sample_rate, ax=ax)
    ax.set(title='Waveform', xlabel='Time (s)', ylabel='Amplitude')
    st.pyplot(fig)

def show_spectrogram(audio, sample_rate):
    stft = librosa.stft(audio)
    spectrogram = librosa.amplitude_to_db(abs(stft), ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(spectrogram, sr=sample_rate, x_axis='time', y_axis='log', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Spectrogram', xlabel='Time (s)', ylabel='Frequency (Hz)')
    st.pyplot(fig)

def show_mel_spectrogram(audio, sample_rate):
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sample_rate, n_mels=128, fmax=8000)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mel_spectrogram_db, sr=sample_rate, x_axis='time', y_axis='mel', fmax=8000, cmap='magma', ax=ax)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')
    ax.set(title='Mel Spectrogram', xlabel='Time (s)', ylabel='Frequency (Hz)')
    st.pyplot(fig)

def show_mfcc(audio, sample_rate):
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=13)
    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(mfccs, sr=sample_rate, x_axis='time', ax=ax)
    fig.colorbar(img, ax=ax)
    ax.set(title='MFCC', xlabel='Time (s)')
    st.pyplot(fig)

def add_noise(waveform, noise_level=0.02):
    noise = tf.random.normal(shape=tf.shape(waveform), mean=0.0, stddev=noise_level)
    return waveform + noise

def adjust_volume(waveform, volume_factor=1.0):
    return waveform * volume_factor

def apply_time_shift(waveform, shift_seconds=0.0, sample_rate=44100):
    shift_samples = int(shift_seconds * sample_rate)
    return tf.roll(waveform, shift_samples, axis=0)

def stretch_signal(waveform, stretch_rate=1.0):
    def _np_stretch(wav_np, rate_np):
        orig_len = wav_np.shape[0]
        target_len = int(orig_len / rate_np)
        x_old = np.linspace(0, orig_len-1, orig_len)
        x_new = np.linspace(0, orig_len-1, target_len)
        stretched = np.interp(x_new, x_old, wav_np).astype(np.float32)
        if stretched.shape[0] < orig_len:
            stretched = np.pad(stretched, (0, orig_len - stretched.shape[0]), mode='constant')
        else:
            stretched = stretched[:orig_len]
        return stretched
    stretched = tf.numpy_function(_np_stretch, [waveform, stretch_rate], tf.float32)
    stretched.set_shape(waveform.shape)
    return stretched

TARGET_SR = 16000
RECORD_DURATION = 1.0
commands = load_commands()

REPRESENTATIONS = {
    'Spectrogram': 'SpectModels',
    'Mel Spectrogram': 'MelModels',
    'MFCC': 'MFCCModels'
}

selected_rep = st.sidebar.selectbox("Data Representation", list(REPRESENTATIONS.keys()))
MODEL_DIR = os.path.join('models', REPRESENTATIONS[selected_rep])

with st.sidebar.expander("Audio Augmentations", expanded=True):
    st.markdown("### Noise Settings")
    noise_level = st.slider("Noise Level", 0.0, 0.1, 0.0, step=0.001, format="%.3f")
    
    st.markdown("### Volume Settings")
    volume_factor = st.slider("Volume Factor", 0.1, 3.0, 1.0, step=0.1)
    
    st.markdown("### Time Shift")
    time_shift = st.slider("Shift Seconds", -1.0, 1.0, 0.0, step=0.01)
    
    st.markdown("### Time Stretch")
    stretch_rate = st.slider("Stretch Rate", 0.5, 2.0, 1.0, step=0.01)

@st.cache_resource
def load_model(path: str) -> tf.keras.Model:
    return tf.keras.models.load_model(path)

def list_model_files(directory: str):
    return [f for f in os.listdir(directory) if f.endswith(('.keras', '.h5'))]

model_files = list_model_files(MODEL_DIR)
if not model_files:
    st.error(f"No models found in '{MODEL_DIR}' directory!")
    st.stop()

selected_model = st.sidebar.selectbox("Model Selection", model_files)
model_path = os.path.join(MODEL_DIR, selected_model)
model = load_model(model_path)

@st.cache_data
def is_silent(audio, threshold=0.04):
    rms = np.sqrt(np.mean(audio**2))
    return rms < threshold

@st.cache_data
def get_spectrogram(waveform: np.ndarray) -> tf.Tensor:
    waveform = waveform[:TARGET_SR]
    zero_padding = tf.zeros([TARGET_SR] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.cast(waveform, dtype=tf.float32)
    equal_length = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(equal_length, frame_length=255, frame_step=128)
    spectrogram = tf.abs(spectrogram)[..., tf.newaxis]
    return spectrogram

@st.cache_data
def get_mel_spectrogram(waveform: np.ndarray) -> tf.Tensor:
    waveform = waveform[:TARGET_SR]
    zero_padding = tf.zeros([TARGET_SR] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    spectrogram = tf.signal.stft(waveform, frame_length=400, frame_step=160)
    spectrogram = tf.abs(spectrogram)
    num_mel_bins = 80
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], TARGET_SR, 80.0, 7600.0
    )
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel, 1)
    mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)[..., tf.newaxis]
    return mel_spectrogram

@st.cache_data
def get_mfcc(waveform: np.ndarray) -> tf.Tensor:
    waveform = waveform[:TARGET_SR]
    zero_padding = tf.zeros([TARGET_SR] - tf.shape(waveform), dtype=tf.float32)
    waveform = tf.concat([waveform, zero_padding], 0)
    stft = tf.signal.stft(waveform, frame_length=400, frame_step=160)
    spectrogram = tf.abs(stft)
    num_mel_bins = 80
    linear_to_mel = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, spectrogram.shape[-1], TARGET_SR, 80.0, 7600.0
    )
    mel = tf.tensordot(spectrogram, linear_to_mel, 1)
    log_mel = tf.math.log(mel + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel)[..., :13]
    return mfccs[..., tf.newaxis]

FEATURE_FUNCS = {
    'Spectrogram': get_spectrogram,
    'Mel Spectrogram': get_mel_spectrogram,
    'MFCC': get_mfcc
}

@st.cache_data
def preprocess(waveform: np.ndarray, method: str) -> tf.Tensor:
    return FEATURE_FUNCS[method](waveform)

def predict_command(model, waveform: np.ndarray, method: str):
    features = preprocess(waveform, method)
    features = tf.expand_dims(features, 0)
    logits = model(features, training=False)
    probs = tf.nn.softmax(logits, axis=-1).numpy()[0]
    idx = np.argmax(probs)
    return idx, probs[idx]

def apply_augmentations(waveform_np, sample_rate):
    waveform = tf.convert_to_tensor(waveform_np, dtype=tf.float32)
    
    if noise_level > 0:
        waveform = add_noise(waveform, noise_level)
    
    if volume_factor != 1.0:
        waveform = adjust_volume(waveform, volume_factor)
    
    if time_shift != 0.0:
        waveform = apply_time_shift(waveform, time_shift, sample_rate)
    
    if stretch_rate != 1.0:
        waveform = stretch_signal(waveform, stretch_rate)
    
    return waveform.numpy()

if 'recorded_audio' not in st.session_state:
    st.session_state['recorded_audio'] = None
    st.session_state['augmented_audio'] = None
    st.session_state['sample_rate'] = 44100

if st.button('Record Command'):
    try:
        device_info = sd.query_devices(None, 'input')
        sample_rate = int(device_info['default_samplerate'])
        st.session_state['sample_rate'] = sample_rate
    except Exception as e:
        st.error(f"Error getting input device: {e}")
        sample_rate = st.session_state['sample_rate']
    
    st.write(f"Recording for {RECORD_DURATION:.1f} seconds at {sample_rate}Hz...")
    audio = sd.rec(int(RECORD_DURATION * sample_rate), 
                  samplerate=sample_rate, 
                  channels=1, 
                  dtype='float32')
    sd.wait()
    audio = audio.flatten()
    st.session_state['recorded_audio'] = audio
    st.session_state['augmented_audio'] = None
    st.success("Audio recorded!")

if st.session_state['recorded_audio'] is not None:
    current_sample_rate = st.session_state['sample_rate']
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Original Audio:**")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, st.session_state.recorded_audio, current_sample_rate)
            st.audio(tmpfile.name)
    
    with col2:
        st.write("**Augmented Audio:**")
        augmented = apply_augmentations(st.session_state.recorded_audio, current_sample_rate)
        st.session_state.augmented_audio = augmented
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            sf.write(tmpfile.name, augmented, current_sample_rate)
            st.audio(tmpfile.name)

    if st.button('Show Visualizations'):
        st.write("### Original Audio Analysis")
        show_waveform(st.session_state.recorded_audio, current_sample_rate)
        show_spectrogram(st.session_state.recorded_audio, current_sample_rate)
        show_mel_spectrogram(st.session_state.recorded_audio, current_sample_rate)
        show_mfcc(st.session_state.recorded_audio, current_sample_rate)
        
        st.write("### Augmented Audio Analysis")
        show_waveform(st.session_state.augmented_audio, current_sample_rate)
        show_spectrogram(st.session_state.augmented_audio, current_sample_rate)
        show_mel_spectrogram(st.session_state.augmented_audio, current_sample_rate)
        show_mfcc(st.session_state.augmented_audio, current_sample_rate)

    if st.button('Classify Command'):
        audio_to_classify = st.session_state.augmented_audio
        current_sample_rate = st.session_state['sample_rate']
        
        if is_silent(audio_to_classify):
            st.warning("No significant audio detected!")
        else:
            audio_resampled = librosa.resample(audio_to_classify, 
                                              orig_sr=current_sample_rate, 
                                              target_sr=TARGET_SR)
            start_time = time.time()
            cmd_id, confidence = predict_command(model, audio_resampled, selected_rep)
            elapsed = time.time() - start_time
            
            st.write("## Classification Results")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(label="Predicted Command", value=commands[cmd_id])
            with col2:
                st.metric(label="Confidence", value=f"{confidence * 100:.1f}%")
            with col3:
                st.metric(label="Latency", value=f"{elapsed * 1000:.1f} ms")

st.write("Click 'Record Command' to start. Adjust parameters in the sidebar.")
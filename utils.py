
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
import random
import os
import yaml
import torch.nn as nn
import scipy
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import parselmouth
from parselmouth.praat import call
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, Gain
import wandb


class ClinicalFeatureExtractor:
    def __init__(self, sr=16000):
        self.sr = sr
        
    def extract_prosodic_features(self, waveform):
        """Extract prosodic features using Parselmouth (Praat integration)"""
        try:
            # Convert torch tensor to numpy if needed
            if hasattr(waveform, 'numpy'):
                audio_np = waveform.numpy()
            else:
                audio_np = waveform
                
            # Create Parselmouth Sound object
            sound = parselmouth.Sound(audio_np, sampling_frequency=self.sr)
            
            # F0 extraction
            pitch = sound.to_pitch(time_step=0.01, pitch_floor=75, pitch_ceiling=300)  # 75-300 Hz range for speech
            f0_values = pitch.selected_array['frequency']
            f0_values = f0_values[f0_values != 0] # Remove unvoiced frames
            #print(f0_values)
            
            # Intensity extraction
            intensity = sound.to_intensity(time_step=0.01, minimum_pitch=75.0)
            intensity_values = intensity.values[0]
            #print(f"intensity {intensity_values}")
            
            prosodic_features = {}
            
            if len(f0_values) > 0:
                prosodic_features.update({
                    'f0_mean': np.mean(f0_values),
                    'f0_std': np.std(f0_values),
                    'f0_range': np.max(f0_values) - np.min(f0_values),
                    'f0_cv': np.std(f0_values) / np.mean(f0_values) if np.mean(f0_values) > 0 else 0,
                    'f0_slope': self._calculate_f0_slope(f0_values),
                    'voiced_frames_ratio': len(f0_values) /pitch.get_number_of_frames()

                })
            else:
                prosodic_features.update({
                    'f0_mean': 0, 'f0_std': 0, 'f0_range': 0, 
                    'f0_cv': 0, 'f0_slope': 0, 'voiced_frames_ratio': 0
                })
            
            if len(intensity_values) > 0:
                prosodic_features.update({
                    'intensity_mean': np.mean(intensity_values),
                    'intensity_std': np.std(intensity_values),
                    'intensity_range': np.max(intensity_values) - np.min(intensity_values)
                })
            else:
                prosodic_features.update({
                    'intensity_mean': 0, 'intensity_std': 0, 'intensity_range': 0
                })
                
            return prosodic_features
            
        except Exception as e:
            print(f"Error in prosodic feature extraction: {e}")
            # Return zero features if extraction fails
            return {
                'f0_mean': 0, 'f0_std': 0, 'f0_range': 0, 'f0_cv': 0, 'f0_slope': 0,
                'voiced_frames_ratio': 0, 'intensity_mean': 0, 'intensity_std': 0, 'intensity_range': 0
            }
    
    def _calculate_f0_slope(self, f0_values):
        """Calculate F0 slope using linear regression"""
        if len(f0_values) < 2:
            return 0
        x = np.arange(len(f0_values))
        slope, _, _, _, _ = scipy.stats.linregress(x, f0_values)
        return slope
    
    def extract_voice_quality_features(self, waveform):
        """Extract voice quality features"""
        try:
            if hasattr(waveform, 'numpy'):
                audio_np = waveform.numpy()
            else:
                audio_np = waveform
                
            sound = parselmouth.Sound(audio_np, sampling_frequency=self.sr)
            
            # Jitter and Shimmer
            pointprocess = call(sound, "To PointProcess (periodic, cc)", 75, 300)
            jitter = call(pointprocess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer = call([sound, pointprocess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
            
            # Harmonics-to-Noise Ratio
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr_mean = call(harmonicity, "Get mean", 0, 0)
            
            # Spectral measures
            spectrum = call(sound, "To Spectrum", "yes")
            spectral_centroid = call(spectrum, "Get centre of gravity", 2)
            
            return {
                'jitter': jitter if not np.isnan(jitter) else 0,
                'shimmer': shimmer if not np.isnan(shimmer) else 0,
                'hnr_mean': hnr_mean if not np.isnan(hnr_mean) else 0,
                'spectral_centroid': spectral_centroid if not np.isnan(spectral_centroid) else 0
            }
            
        except Exception as e:
            print(f"Error in voice quality feature extraction: {e}")
            return {'jitter': 0, 'shimmer': 0, 'hnr_mean': 0, 'spectral_centroid': 0}
    
    def extract_temporal_features(self, waveform):
        """Extract temporal and fluency features"""
        try:
            if hasattr(waveform, 'numpy'):
                audio_np = waveform.numpy()
            else:
                audio_np = waveform
            
            # Voice activity detection (simple energy-based)
            frame_length = int(0.025 * self.sr)  # 25ms frames
            hop_length = int(0.01 * self.sr)    # 10ms hop
            
            # Calculate energy
            energy = []
            for i in range(0, len(audio_np) - frame_length, hop_length):
                frame = audio_np[i:i + frame_length]
                energy.append(np.sum(frame ** 2))
            
            energy = np.array(energy)
            threshold = np.percentile(energy, 30)  # Adaptive threshold
            voiced_frames = energy > threshold
            
            # Calculate pause statistics
            speech_segments = self._get_speech_segments(voiced_frames, hop_length)
            pause_segments = self._get_pause_segments(voiced_frames, hop_length)
            
            total_duration = len(audio_np) / self.sr
            total_speech_time = sum([seg[1] - seg[0] for seg in speech_segments])
            total_pause_time = sum([seg[1] - seg[0] for seg in pause_segments])
            
            return {
                'speech_rate': total_speech_time / total_duration if total_duration > 0 else 0,
                'pause_rate': len(pause_segments) / total_duration if total_duration > 0 else 0,
                'mean_pause_duration': np.mean([seg[1] - seg[0] for seg in pause_segments]) if pause_segments else 0,
                'speech_to_pause_ratio': total_speech_time / total_pause_time if total_pause_time > 0 else np.inf,
                'voiced_frame_ratio': np.sum(voiced_frames) / len(voiced_frames) if len(voiced_frames) > 0 else 0
            }
            
        except Exception as e:
            print(f"Error in temporal feature extraction: {e}")
            return {
                'speech_rate': 0, 'pause_rate': 0, 'mean_pause_duration': 0, 
                'speech_to_pause_ratio': 0, 'voiced_frame_ratio': 0
            }
    
    def _get_speech_segments(self, voiced_frames, hop_length):
        """Get continuous speech segments"""
        segments = []
        start = None
        
        for i, is_voiced in enumerate(voiced_frames):
            if is_voiced and start is None:
                start = i * hop_length / self.sr
            elif not is_voiced and start is not None:
                end = i * hop_length / self.sr
                segments.append((start, end))
                start = None
                
        if start is not None:
            segments.append((start, len(voiced_frames) * hop_length / self.sr))
            
        return segments
    
    def _get_pause_segments(self, voiced_frames, hop_length):
        """Get pause segments"""
        segments = []
        start = None
        
        for i, is_voiced in enumerate(voiced_frames):
            if not is_voiced and start is None:
                start = i * hop_length / self.sr
            elif is_voiced and start is not None:
                end = i * hop_length / self.sr
                if end - start > 0.1:  # Only count pauses longer than 100ms
                    segments.append((start, end))
                start = None
                
        if start is not None:
            end = len(voiced_frames) * hop_length / self.sr
            if end - start > 0.1:
                segments.append((start, end))
                
        return segments
    
    def extract_all_features(self, waveform):
        """Extract all clinical features"""
        prosodic = self.extract_prosodic_features(waveform)
        voice_quality = self.extract_voice_quality_features(waveform)
        temporal = self.extract_temporal_features(waveform)
        
        # Combine all features
        all_features = {
            **prosodic, **voice_quality, **temporal
                        }
        return all_features


# Fit and apply normalization to clinical features

class ClinicalFeatureNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, dataset):
        # Collect all clinical features from the dataset
        feats = []
        for i in range(len(dataset)):
            item = dataset[i]
            if len(item) == 3:
                _, clinical, _ = item
                feats.append(clinical.numpy())
        feats = np.stack(feats)
        self.mean = feats.mean(axis=0)
        self.std = feats.std(axis=0) + 1e-8  # avoid division by zero

    def transform(self, clinical_tensor):
        # Normalize a single tensor (1D or batched 2D)
        return (clinical_tensor - torch.tensor(self.mean, dtype=clinical_tensor.dtype)) / torch.tensor(self.std, dtype=clinical_tensor.dtype)

    def fit_transform(self, dataset):
        self.fit(dataset)
        return [self.transform(torch.tensor(f, dtype=torch.float32)) for f in dataset]


# model utility

class LockedDropout(nn.Module):
    def __init__(self, p=0.3):
        super().__init__()
        self.p = p
        self.mask = None
        
    def forward(self, x):
        if not self.training or self.p == 0:
            return x
            
        # Create mask if none exists or batch size changes
        if self.mask is None or self.mask.size(0) != x.size(0):
            # (batch_size, 1, hidden_size)
            self.mask = x.new_empty(x.size(0), 1, x.size(2), 
                          requires_grad=False).bernoulli_(1 - self.p) / (1 - self.p)
            
        return self.mask.expand_as(x) * x
    
# Enhanced Model Architecture with Feature Fusion
class EnhancedDementiaCNNBiLSTM(nn.Module):
    def __init__(self, use_clinical_features=True, clinical_feature_dim=18):
        super(EnhancedDementiaCNNBiLSTM, self).__init__()
        self.use_clinical_features = use_clinical_features
        
        # Acoustic feature processing (CNN + BiLSTM)
        self.acoustic_cnn = nn.Sequential(
            nn.Conv1d(70, 128, kernel_size=5, padding=2),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.SiLU()
        )
        
        self.acoustic_lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=1,
            bidirectional=True
        )
        
        self.locked_dropout = LockedDropout(p=0.3)
        
        # Clinical feature processing
        if use_clinical_features:
            self.clinical_processor = nn.Sequential(
                nn.Linear(clinical_feature_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(32, 16),
                nn.ReLU()
            )
            
            # Feature fusion
            self.fusion_layer = nn.Sequential(
                nn.Linear(64 + 16, 64),  # acoustic_features + clinical_features
                nn.ReLU(),
                nn.Dropout(0.4)
            )
            
            # Final classifier
            self.classifier = nn.Sequential(
                nn.Linear(64, 32),
                nn.SiLU(),
                nn.Dropout(0.4),
                nn.Linear(32, 1)
            )
        else:
            # Original classifier for acoustic features only
            self.classifier = nn.Sequential(
                nn.Linear(64, 32),
                nn.SiLU(),
                nn.Dropout(0.4),
                nn.Linear(32, 1)
            )
    
    def forward(self, acoustic_input, clinical_input=None):
        # Process acoustic features
        # acoustic_input shape: (batch_size, seq_len, n_mels)
        batch_size = acoustic_input.size(0)
        x_acoustic = acoustic_input.permute(0, 2, 1)  # (batch, n_mels, seq_len)
        
        # CNN processing
        x_acoustic = self.acoustic_cnn(x_acoustic)  # (batch, 64, seq_len)
        x_acoustic = x_acoustic.permute(0, 2, 1)  # (batch, seq_len, 64)
        
        # BiLSTM processing
        x_acoustic = self.locked_dropout(x_acoustic)
        lstm_out, _ = self.acoustic_lstm(x_acoustic)  # (batch, seq_len, 64)
        
        # Temporal pooling for acoustic features
        acoustic_features = torch.mean(lstm_out, dim=1)  # (batch, 64)
        
        if self.use_clinical_features and clinical_input is not None:
            # Process clinical features
            clinical_features = self.clinical_processor(clinical_input)  # (batch, 16)
            
            # Feature fusion
            combined_features = torch.cat([acoustic_features, clinical_features], dim=1)  # (batch, 80)
            fused_features = self.fusion_layer(combined_features)  # (batch, 64)
            
            # Classification
            output = self.classifier(fused_features)
        else:
            # Use only acoustic features
            output = self.classifier(acoustic_features)
        
        return output
    
    
# inference utils
# Configuration dictionary matching training parameters
config = {
    'sr': 16000,
    'n_mels': 70,
    'chunk_length': 5.0,    # in seconds
    'chunk_overlap': 2.0    # in seconds
}

def load_audio(file_path, target_sr=config['sr']):
    """Load and preprocess audio file"""
    waveform, sample_rate = torchaudio.load(file_path)
    waveform = torch.mean(waveform, dim=0)
    if sample_rate != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform

def extract_fbank(waveform, config):
    """Extract acoustic features (FBank)"""
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0),
        num_mel_bins=config['n_mels'],
        sample_frequency=config['sr']
    )
    fbank = (fbank - fbank.mean(dim=0)) / (fbank.std(dim=0) + 1e-6)
    return fbank

def extract_clinical_features(waveform):
    """Extract clinical features from waveform"""
    clinical_extractor = ClinicalFeatureExtractor()
    clinical_features_dict = clinical_extractor.extract_all_features(waveform)
    
    # Convert to tensor and handle any inf/nan values
    clinical_values = []
    for key in sorted(clinical_features_dict.keys()):  # Ensure consistent ordering
        value = clinical_features_dict[key]
        if np.isinf(value) or np.isnan(value):
            value = 0.0
        clinical_values.append(value)
    
    clinical_features = torch.tensor(clinical_values, dtype=torch.float32)
    return clinical_features

def visualize_fbank(fbank, title="FBank Visualization"):
    """Visualize acoustic features"""
    fbank_np = fbank.cpu().numpy().T
    plt.figure(figsize=(10, 4))
    plt.imshow(fbank_np, aspect='auto', origin='lower', interpolation='nearest')
    plt.title(title)
    plt.xlabel("Time Frame")
    plt.ylabel("Mel Bin")
    plt.colorbar(label='Amplitude')
    plt.tight_layout()
    plt.show()
    
def chunk_fbank(fbank, config):
    """Split acoustic features into chunks"""
    chunk_frames = int(config['chunk_length'] * (config['sr'] / 160))
    overlap_frames = int(config['chunk_overlap'] * (config['sr'] / 160))
    stride = chunk_frames - overlap_frames
    chunks = []
    n_frames = fbank.shape[0]
    for start in range(0, n_frames, stride):
        end = start + chunk_frames
        chunk = fbank[start:end]
        if chunk.shape[0] < chunk_frames:
            pad_size = chunk_frames - chunk.shape[0]
            chunk = F.pad(chunk, (0, 0, 0, pad_size))
        chunks.append(chunk)
    return torch.stack(chunks)

def run_enhanced_inference(model, chunks, clinical_features, device):
    """Run inference with both acoustic and clinical features"""
    model.eval()
    chunks = chunks.to(device)
    
    # Repeat clinical features for each chunk
    num_chunks = len(chunks)
    clinical_features_repeated = clinical_features.unsqueeze(0).repeat(num_chunks, 1).to(device)
    
    with torch.no_grad():
        outputs = model(chunks, clinical_features_repeated)
    
    avg_output = outputs.mean(dim=0)
    probability = torch.sigmoid(avg_output)
    return probability.item()

def enhanced_inference_pipeline(file_path, model, device, config, visualize=False):
    """Complete enhanced inference pipeline"""
    print(f"Processing: {file_path}")
    
    # Load audio
    waveform = load_audio(file_path, target_sr=config['sr'])
    print(f"✓ Audio loaded: {waveform.shape}")
    
    # Extract acoustic features
    fbank = extract_fbank(waveform, config)
    print(f"✓ Acoustic features extracted: {fbank.shape}")
    
    # Extract clinical features
    clinical_features = extract_clinical_features(waveform)
    print(f"✓ Clinical features extracted: {clinical_features.shape}")
    
    # Visualize if requested
    if visualize:
        visualize_fbank(fbank, title="Mel Spectrogram")
    
    # Chunk acoustic features
    chunks = chunk_fbank(fbank, config)
    print(f"✓ Audio chunked: {chunks.shape}")
    
    # Run enhanced inference
    prediction = run_enhanced_inference(model, chunks, clinical_features, device)
    print(f"✓ Prediction: {prediction*100:.2f}% dementia risk")
    
    return prediction


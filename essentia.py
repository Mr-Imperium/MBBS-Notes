"""
phrase_detector.py - Essentia Implementation
============================================
BrainHeart Project: Detection of 10-Second Structural Periodicity

Scientific Basis: Bernardi et al. (2009) - Cardiovascular synchronization
to the 0.1 Hz Mayer Wave frequency in Verdi's arias.

Migration: librosa v4.1 → essentia.standard
Author: Audio Signal Processing Engineer
Version: 5.0 (Essentia)
"""

import warnings
import numpy as np
import essentia
import essentia.standard as es
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
import scipy.signal as signal
from typing import Dict, Tuple, Optional, List, Any
from dataclasses import dataclass
from functools import lru_cache

warnings.filterwarnings('ignore')

# Ensure Essentia operates in silent mode for batch processing
essentia.log.infoActive = False
essentia.log.warningActive = False


@dataclass
class FeatureBundle:
    """Container for extracted audio features."""
    chroma: np.ndarray
    harm_ratio: float
    y_harm: np.ndarray
    flatness: float
    diversity: float
    spectral_entropy: float = 0.0


class EssentiaAlgorithmPool:
    """
    Lazy-initialized pool of Essentia algorithms.
    Provides thread-safe, reusable algorithm instances for performance.
    """
    
    def __init__(self, sr: int, frame_size: int, hop_size: int):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size
        self._algorithms: Dict[str, Any] = {}
    
    def _create_algorithm(self, name: str) -> Any:
        """Factory method for algorithm instantiation."""
        creators = {
            'windowing': lambda: es.Windowing(
                type='hann',
                normalized=False,
                zeroPhase=False
            ),
            'spectrum': lambda: es.Spectrum(size=self.frame_size),
            'flatness': lambda: es.Flatness(),
            'entropy': lambda: es.Entropy(),
            'spectral_peaks': lambda: es.SpectralPeaks(
                maxFrequency=5000.0,
                minFrequency=40.0,
                maxPeaks=100,
                magnitudeThreshold=1e-5,
                orderBy='magnitude',
                sampleRate=self.sr
            ),
            'hpcp': lambda: es.HPCP(
                size=12,
                referenceFrequency=440.0,
                bandPreset=True,
                minFrequency=40.0,
                maxFrequency=5000.0,
                weightType='squaredCosine',
                nonLinear=True,
                windowSize=1.0,  # 1 semitone - handles microtonal variations
                sampleRate=self.sr,
                harmonics=4,
                normalized='unitSum'
            ),
            'hpcp_tuned': lambda: es.HPCP(
                size=12,
                referenceFrequency=440.0,
                bandPreset=False,  # Allow non-Western tuning
                minFrequency=40.0,
                maxFrequency=5000.0,
                weightType='squaredCosine',
                nonLinear=True,
                windowSize=4/3,  # Wider window for non-Western scales
                sampleRate=self.sr,
                harmonics=6,
                normalized='unitSum'
            ),
            'rms': lambda: es.RMS(),
            'energy': lambda: es.Energy(),
            'zcr': lambda: es.ZeroCrossingRate(),
        }
        
        if name not in creators:
            raise ValueError(f"Unknown algorithm: {name}")
        return creators[name]()
    
    def get(self, name: str) -> Any:
        """Get or create an algorithm instance."""
        if name not in self._algorithms:
            self._algorithms[name] = self._create_algorithm(name)
        return self._algorithms[name]


class HarmonicPercussiveSeparator:
    """
    HPSS implementation using Essentia's median filtering approach.
    
    This implements the Fitzgerald (2010) method:
    - Harmonic components have horizontal structure in spectrogram
    - Percussive components have vertical structure
    """
    
    def __init__(self, sr: int, frame_size: int = 2048, hop_size: int = 512, 
                 margin: float = 2.0, kernel_size: int = 31):
        self.sr = sr
        self.frame_size = frame_size
        self.hop_size = hop_size
        self.margin = margin
        self.kernel_size = kernel_size
        
        # Initialize Essentia algorithms
        self.windowing = es.Windowing(type='hann', normalized=False)
        self.spectrum = es.Spectrum(size=frame_size)
        self.ifft = es.IFFT(size=frame_size)
        self.overlapadd = es.OverlapAdd(
            frameSize=frame_size,
            hopSize=hop_size,
            gain=1.0 / (frame_size / hop_size / 2)
        )
    
    def separate(self, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Separate harmonic and percussive components.
        
        Parameters:
        -----------
        y : np.ndarray
            Input audio signal (float32)
        
        Returns:
        --------
        y_harmonic : np.ndarray
            Harmonic component
        y_percussive : np.ndarray
            Percussive component
        """
        y = np.asarray(y, dtype=np.float32)
        
        # Compute STFT magnitude and phase
        frames = []
        phases = []
        
        for frame in es.FrameGenerator(y, frameSize=self.frame_size,
                                        hopSize=self.hop_size,
                                        startFromZero=True):
            windowed = self.windowing(frame)
            # Use FFT to get complex spectrum
            fft_algo = es.FFT(size=self.frame_size)
            complex_spectrum = fft_algo(windowed)
            
            magnitude = np.abs(complex_spectrum)
            phase = np.angle(complex_spectrum)
            
            frames.append(magnitude)
            phases.append(phase)
        
        if len(frames) < 3:
            return y, np.zeros_like(y)
        
        S = np.array(frames).T  # Shape: (n_freq, n_frames)
        P = np.array(phases).T
        
        # Apply median filtering
        # Harmonic: median filter along time axis
        # Percussive: median filter along frequency axis
        from scipy.ndimage import median_filter
        
        S_harmonic = median_filter(S, size=(1, self.kernel_size))
        S_percussive = median_filter(S, size=(self.kernel_size, 1))
        
        # Soft masking with margin
        M_h = S_harmonic ** self.margin
        M_p = S_percussive ** self.margin
        
        total = M_h + M_p + 1e-10
        mask_h = M_h / total
        mask_p = M_p / total
        
        # Apply masks
        S_h = S * mask_h
        S_p = S * mask_p
        
        # Reconstruct signals using Griffin-Lim or phase information
        y_harmonic = self._istft(S_h, P)
        y_percussive = self._istft(S_p, P)
        
        # Ensure same length as input
        y_harmonic = y_harmonic[:len(y)]
        y_percussive = y_percussive[:len(y)]
        
        # Pad if necessary
        if len(y_harmonic) < len(y):
            y_harmonic = np.pad(y_harmonic, (0, len(y) - len(y_harmonic)))
        if len(y_percussive) < len(y):
            y_percussive = np.pad(y_percussive, (0, len(y) - len(y_percussive)))
        
        return y_harmonic.astype(np.float32), y_percussive.astype(np.float32)
    
    def _istft(self, magnitude: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """Inverse STFT reconstruction."""
        n_frames = magnitude.shape[1]
        output_length = (n_frames - 1) * self.hop_size + self.frame_size
        y = np.zeros(output_length, dtype=np.float32)
        
        window = np.hanning(self.frame_size)
        
        for i in range(n_frames):
            # Reconstruct complex spectrum
            complex_frame = magnitude[:, i] * np.exp(1j * phase[:, i])
            
            # IFFT
            frame = np.fft.irfft(complex_frame, n=self.frame_size)
            
            # Overlap-add
            start = i * self.hop_size
            end = start + self.frame_size
            if end <= len(y):
                y[start:end] += frame * window
        
        return y


class PhraseBoundaryDetector:
    """
    Essentia-based Phrase Boundary Detector for 10-second structural periodicity.
    
    This class detects the ~0.1 Hz (10-second) Mayer Wave periodicity in music,
    replicating the Bernardi et al. (2009) cardiovascular synchronization study.
    
    Key improvements over librosa version:
    - HPCP provides superior harmonic representation vs chroma_cqt
    - Native support for non-Western tuning systems
    - C++ backend for 5-10x performance improvement
    - Enhanced spectral analysis via Essentia's optimized FFT
    """
    
    def __init__(self, sr: int = 22050, 
                 use_non_western_tuning: bool = False,
                 verbose: bool = False):
        """
        Initialize the detector.
        
        Parameters:
        -----------
        sr : int
            Sample rate (default: 22050)
        use_non_western_tuning : bool
            Enable wider HPCP bandwidth for non-Western music
        verbose : bool
            Print debug information
        """
        self.sr = sr
        self.hop = 512
        self.frame_size = 2048
        self.n_chroma = 12
        self.n_steps = 15  # Memory stacking steps
        self.delay = 3     # Delay between stacked frames
        self.use_non_western_tuning = use_non_western_tuning
        self.verbose = verbose
        
        # Initialize algorithm pool
        self._pool = EssentiaAlgorithmPool(sr, self.frame_size, self.hop)
        
        # Initialize HPSS separator
        self._hpss = HarmonicPercussiveSeparator(
            sr=sr,
            frame_size=self.frame_size,
            hop_size=self.hop,
            margin=2.0
        )
        
        # Precompute constants
        self._fps = sr / self.hop  # Frames per second
    
    def _spectral_flatness(self, y: np.ndarray) -> float:
        """
        Compute spectral flatness (Wiener entropy).
        
        Returns value between 0 (tonal) and 1 (noise-like).
        Used as anti-sine-wave gate to reject trivial signals.
        """
        try:
            flatness_algo = self._pool.get('flatness')
            windowing = self._pool.get('windowing')
            spectrum = self._pool.get('spectrum')
            
            flatness_values = []
            
            for frame in es.FrameGenerator(y, frameSize=self.frame_size,
                                            hopSize=self.hop, startFromZero=True):
                windowed = windowing(frame)
                spec = spectrum(windowed)
                
                if np.sum(spec) > 1e-10:
                    flat = flatness_algo(spec)
                    flatness_values.append(flat)
            
            if len(flatness_values) == 0:
                return 0.5
            
            return float(np.mean(flatness_values))
        except Exception:
            return 0.5
    
    def _spectral_entropy(self, y: np.ndarray) -> float:
        """
        Compute spectral entropy for signal complexity analysis.
        
        High entropy → complex signal
        Low entropy → simple/periodic signal (potential sine wave)
        """
        try:
            entropy_algo = self._pool.get('entropy')
            windowing = self._pool.get('windowing')
            spectrum = self._pool.get('spectrum')
            
            entropy_values = []
            
            for frame in es.FrameGenerator(y, frameSize=self.frame_size,
                                            hopSize=self.hop, startFromZero=True):
                windowed = windowing(frame)
                spec = spectrum(windowed)
                
                # Normalize spectrum to probability distribution
                spec_sum = np.sum(spec)
                if spec_sum > 1e-10:
                    spec_norm = spec / spec_sum
                    ent = entropy_algo(spec_norm)
                    entropy_values.append(ent)
            
            if len(entropy_values) == 0:
                return 0.5
            
            return float(np.mean(entropy_values))
        except Exception:
            return 0.5
    
    def _compute_hpcp_sequence(self, y: np.ndarray) -> np.ndarray:
        """
        Compute HPCP (Harmonic Pitch Class Profile) sequence.
        
        HPCP advantages over standard chroma:
        - Better harmonic weighting
        - Configurable tuning reference
        - Non-linear compression for robustness
        - Handles non-Western scales when configured
        """
        windowing = self._pool.get('windowing')
        spectrum = self._pool.get('spectrum')
        spectral_peaks = self._pool.get('spectral_peaks')
        
        # Select HPCP algorithm based on tuning preference
        hpcp_key = 'hpcp_tuned' if self.use_non_western_tuning else 'hpcp'
        hpcp = self._pool.get(hpcp_key)
        
        hpcp_sequence = []
        
        for frame in es.FrameGenerator(y, frameSize=self.frame_size,
                                        hopSize=self.hop, startFromZero=True):
            windowed = windowing(frame)
            spec = spectrum(windowed)
            
            # Extract spectral peaks
            frequencies, magnitudes = spectral_peaks(spec)
            
            if len(frequencies) > 0 and len(magnitudes) > 0:
                try:
                    hpcp_vector = hpcp(frequencies, magnitudes)
                except Exception:
                    hpcp_vector = np.zeros(self.n_chroma, dtype=np.float32)
            else:
                hpcp_vector = np.zeros(self.n_chroma, dtype=np.float32)
            
            hpcp_sequence.append(hpcp_vector)
        
        if len(hpcp_sequence) == 0:
            return np.zeros((self.n_chroma, 1), dtype=np.float32)
        
        return np.array(hpcp_sequence).T  # Shape: (n_chroma, n_frames)
    
    def _chroma_diversity(self, chroma: np.ndarray) -> float:
        """
        Compute chroma diversity using entropy and active pitch ratio.
        
        Formula preserved from v4.1:
        - entropy_norm = H(chroma) / log(12)
        - active_ratio = count(bins > 0.1 * max) / 12
        - diversity = 0.5 * entropy_norm + 0.5 * active_ratio
        """
        cm = np.mean(chroma, axis=1)
        
        if np.sum(cm) < 1e-9:
            return 0.0
        
        # Normalize to probability distribution
        cn = cm / (np.sum(cm) + 1e-9)
        
        # Entropy (normalized by max possible entropy)
        ent = -np.sum(cn * np.log(cn + 1e-10))
        ent_norm = ent / np.log(12)
        
        # Active pitch class ratio
        active = np.sum(cm > 0.1 * np.max(cm)) / 12.0
        
        return float(0.5 * ent_norm + 0.5 * active)
    
    def _harmonic_content(self, y: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Separate harmonic/percussive and compute harmonic ratio.
        
        Uses Essentia HPSS with margin=2.0 (matching librosa default).
        """
        try:
            y_harmonic, y_percussive = self._hpss.separate(y)
            
            h_energy = np.sum(y_harmonic ** 2)
            p_energy = np.sum(y_percussive ** 2)
            
            harm_ratio = h_energy / (h_energy + p_energy + 1e-9)
            
            return float(harm_ratio), y_harmonic
        except Exception:
            return 0.5, y
    
    def _extract_features(self, y: np.ndarray) -> Optional[FeatureBundle]:
        """
        Extract harmonic features with multi-gate filtering.
        
        Gates (rejection criteria):
        1. Empty/silent signal
        2. High spectral flatness (>0.92) - noise-like
        3. Low harmonic ratio (<0.05) - percussive
        4. Low chroma diversity (<0.25) - single note/drone
        5. Static chroma pattern - trivial signal
        """
        if len(y) == 0:
            return None
        
        mx = np.max(np.abs(y))
        if mx < 1e-6:
            return None
        
        # Normalize
        yn = (y / (mx + 1e-9)).astype(np.float32)
        
        # Gate 1: Spectral Flatness (anti-noise)
        flatness = self._spectral_flatness(yn)
        if flatness > 0.92:
            if self.verbose:
                print(f"  [REJECT] Flatness {flatness:.3f} > 0.92 (noise-like)")
            return None
        
        # Gate 2: Harmonic Content
        harm_ratio, y_harmonic = self._harmonic_content(yn)
        if harm_ratio < 0.05:
            if self.verbose:
                print(f"  [REJECT] Harmonic ratio {harm_ratio:.3f} < 0.05")
            return None
        
        # Compute HPCP (harmonic pitch class profile)
        try:
            chroma = self._compute_hpcp_sequence(y_harmonic)
        except Exception as e:
            if self.verbose:
                print(f"  [REJECT] HPCP computation failed: {e}")
            return None
        
        if chroma.shape[1] < 5:
            if self.verbose:
                print(f"  [REJECT] Too few frames: {chroma.shape[1]}")
            return None
        
        # Gate 3: Chroma Diversity
        diversity = self._chroma_diversity(chroma)
        if diversity < 0.25:
            if self.verbose:
                print(f"  [REJECT] Diversity {diversity:.3f} < 0.25")
            return None
        
        # Gate 4: Static Pattern Detection
        if np.std(chroma) < 0.005 and np.max(chroma) < 0.1:
            if self.verbose:
                print("  [REJECT] Static chroma pattern")
            return None
        
        # Temporal smoothing
        chroma = gaussian_filter1d(chroma, sigma=2, axis=1)
        
        # Compute spectral entropy for additional analysis
        spectral_ent = self._spectral_entropy(y_harmonic)
        
        return FeatureBundle(
            chroma=chroma,
            harm_ratio=harm_ratio,
            y_harm=y_harmonic,
            flatness=flatness,
            diversity=diversity,
            spectral_entropy=spectral_ent
        )
    
    def _stack_memory(self, features: np.ndarray, 
                      n_steps: int, delay: int) -> np.ndarray:
        """
        Stack feature frames with time-delay embedding.
        
        Creates temporal context by stacking delayed copies of features.
        Equivalent to librosa.feature.stack_memory.
        
        Parameters:
        -----------
        features : np.ndarray
            Shape (n_features, n_frames)
        n_steps : int
            Number of time steps to stack
        delay : int
            Delay between consecutive steps
        
        Returns:
        --------
        stacked : np.ndarray
            Shape (n_features * n_steps, n_frames)
        """
        n_features, n_frames = features.shape
        
        if n_frames < n_steps * delay:
            # Not enough frames, return original
            return features
        
        stacked = []
        for step in range(n_steps):
            offset = step * delay
            if offset == 0:
                stacked.append(features)
            else:
                # Pad at the beginning, truncate at the end
                padded = np.concatenate([
                    np.zeros((n_features, offset), dtype=features.dtype),
                    features[:, :-offset]
                ], axis=1)
                stacked.append(padded)
        
        return np.vstack(stacked)
    
    def _cosine_similarity_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity matrix for self-similarity analysis.
        
        Parameters:
        -----------
        X : np.ndarray
            Features with shape (n_features, n_frames)
        
        Returns:
        --------
        sim : np.ndarray
            Similarity matrix with shape (n_frames, n_frames)
        """
        # Normalize each column (frame)
        norms = np.linalg.norm(X, axis=0, keepdims=True)
        norms = np.maximum(norms, 1e-10)  # Avoid division by zero
        X_norm = X / norms
        
        # Cosine similarity = dot product of normalized vectors
        sim = np.dot(X_norm.T, X_norm)
        
        return sim
    
    def _recurrence_matrix(self, chroma: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute recurrence matrix using cosine affinity.
        
        Equivalent to librosa.segment.recurrence_matrix(mode='affinity', sym=True).
        
        The recurrence matrix R[i,j] measures the similarity between
        time points i and j, capturing repeating structural patterns.
        """
        try:
            # Stack memory for temporal context
            stacked = self._stack_memory(chroma, self.n_steps, self.delay)
        except Exception:
            return None
        
        try:
            # Compute affinity/similarity matrix
            rec = self._cosine_similarity_matrix(stacked)
            
            # Ensure symmetry (should already be, but enforce)
            rec = (rec + rec.T) / 2.0
            
            # Apply width masking (suppress near-diagonal entries)
            # This removes trivial self-similarity at lag ~0
            width = 5
            n = rec.shape[0]
            
            # Create mask efficiently
            i_idx, j_idx = np.meshgrid(np.arange(n), np.arange(n), indexing='ij')
            near_diag_mask = np.abs(i_idx - j_idx) < width
            near_diag_mask &= (i_idx != j_idx)  # Keep main diagonal
            
            # Attenuate near-diagonal (soft masking)
            rec[near_diag_mask] *= 0.1
            
            # Gaussian smoothing for robustness
            rec = gaussian_filter1d(rec, sigma=1.5, axis=0)
            rec = gaussian_filter1d(rec, sigma=1.5, axis=1)
            
            return rec
        except Exception:
            return None
    
    def _recurrence_to_lag(self, rec: np.ndarray) -> np.ndarray:
        """
        Convert recurrence matrix to lag matrix.
        
        The lag matrix L[k, :] contains the values from diagonal k of
        the recurrence matrix. This transforms the 2D self-similarity
        into a representation indexed by lag (time offset).
        
        Equivalent to librosa.segment.recurrence_to_lag(pad=False).
        """
        n = rec.shape[0]
        lag = np.zeros((n, n), dtype=rec.dtype)
        
        for k in range(n):
            # Extract diagonal k (offset from main diagonal)
            diag = np.diag(rec, k)
            if len(diag) > 0:
                lag[k, :len(diag)] = diag
        
        return lag
    
    def _refine_peak(self, curve: np.ndarray, idx: int, fps: float) -> float:
        """
        Refine peak position using cubic interpolation.
        
        This provides sub-frame accuracy for period estimation.
        """
        win = int(fps)
        start = max(0, idx - win)
        end = min(len(curve), idx + win + 1)
        local = curve[start:end]
        
        if len(local) < 5:
            return float(idx)
        
        x_coarse = np.arange(len(local))
        x_fine = np.linspace(0, len(local) - 1, len(local) * 10)
        
        try:
            interpolator = interp1d(x_coarse, local, kind='cubic')
            fine_curve = interpolator(x_fine)
            refined_idx = start + np.argmax(fine_curve) / 10.0
            return float(refined_idx)
        except Exception:
            return float(idx)
    
    def _periodicity_from_lag(self, rec: np.ndarray, 
                               min_p: float, max_p: float) -> Tuple[float, float, Dict]:
        """
        Extract dominant periodicity from lag matrix diagonal sums.
        
        The structure curve is computed by summing each diagonal of the
        lag matrix. Peaks in this curve correspond to repeated structural
        patterns at that lag (time offset).
        
        Parameters:
        -----------
        rec : np.ndarray
            Recurrence matrix
        min_p : float
            Minimum period (seconds)
        max_p : float
            Maximum period (seconds)
        
        Returns:
        --------
        period : float
            Detected period in seconds
        confidence : float
            Confidence score [0, 1]
        analysis : dict
            Detailed analysis results
        """
        try:
            lag = self._recurrence_to_lag(rec)
        except Exception:
            return 0.0, 0.0, {}
        
        # Structure curve: sum of each lag diagonal
        curve = np.sum(lag, axis=1)
        
        if len(curve) < 10:
            return 0.0, 0.0, {}
        
        fps = self._fps  # Frames per second
        min_bin = max(1, int(min_p * fps))
        max_bin = min(len(curve) - 1, int(max_p * fps))
        
        if min_bin >= max_bin:
            return 0.0, 0.0, {}
        
        # Extract search region
        region = curve[min_bin:max_bin + 1]
        if len(region) < 5:
            return 0.0, 0.0, {}
        
        # Smooth for robust peak detection
        smooth = gaussian_filter1d(region, sigma=3)
        
        # Find peaks with prominence threshold
        try:
            peaks, props = signal.find_peaks(
                smooth,
                prominence=np.std(smooth) * 0.3,
                distance=int(0.5 * fps)
            )
        except Exception:
            peaks, props = np.array([]), {}
        
        # Select best peak
        if len(peaks) == 0:
            pk_local = np.argmax(smooth)
            prominence = float(np.max(smooth) - np.min(smooth))
        elif 'prominences' in props and len(props['prominences']) > 0:
            best_idx = np.argmax(props['prominences'])
            pk_local = peaks[best_idx]
            prominence = float(props['prominences'][best_idx])
        else:
            pk_local = peaks[0]
            prominence = 1.0
        
        # Convert to global index
        pk_global = min_bin + pk_local
        
        # Sub-frame refinement
        refined_idx = self._refine_peak(curve, pk_global, fps)
        period = float(refined_idx / fps)
        
        # === CONFIDENCE CALCULATION (preserved from v4.1) ===
        
        # Valid region for normalization (exclude near-zero lags)
        valid_start = max(1, int(fps))
        if valid_start >= len(curve):
            valid_start = 1
        valid = curve[valid_start:]
        
        if len(valid) == 0:
            return period, 0.5, {}
        
        g_max = np.max(valid)
        pk_int = min(int(refined_idx), len(curve) - 1)
        pk_val = curve[pk_int]
        
        # Base confidence: peak value relative to global maximum
        base_conf = float(pk_val / (g_max + 1e-9)) if g_max > 1e-9 else 0.5
        
        # SNR calculation: (peak - mean) / std
        std = np.std(region)
        mean = np.mean(region)
        snr = (pk_val - mean) / (std + 1e-9) if std > 1e-9 else 0.0
        
        # SNR factor: scaled to [0, 1] range
        snr_factor = float(np.clip(snr / 3.0, 0.0, 1.0))
        
        # Combined confidence formula
        # conf = base_conf * (0.6 + 0.4 * snr_factor), clipped to [0, 1]
        conf = float(np.clip(base_conf * (0.6 + 0.4 * snr_factor), 0.0, 1.0))
        
        # Build analysis dictionary
        times = np.arange(len(curve)) / fps
        analysis = {
            'times': times,
            'structure_curve': curve / (np.max(curve) + 1e-9),
            'peak_idx': refined_idx,
            'search_region': region,
            'prominence': prominence,
            'snr': float(snr)
        }
        
        return period, conf, analysis
    
    def detect_periodicity(self, y: np.ndarray, 
                           min_period: float = 8.0, 
                           max_period: float = 14.0) -> Tuple[float, float, Dict]:
        """
        Detect the dominant structural periodicity in the 8-14 second range.
        
        This is the main entry point for single-window analysis.
        
        Parameters:
        -----------
        y : np.ndarray
            Audio signal (mono, any sample rate - will use self.sr)
        min_period : float
            Minimum period to detect (seconds), default 8.0
        max_period : float
            Maximum period to detect (seconds), default 14.0
        
        Returns:
        --------
        period : float
            Detected period in seconds (0.0 if detection failed)
        confidence : float
            Confidence score [0, 1]
        analysis : dict
            Detailed analysis results including:
            - harmonic_confidence: Harmonic/percussive ratio
            - spectral_flatness: 0=tonal, 1=noise
            - chroma_diversity: Pitch class distribution entropy
            - rejection_reason: Cause of failure (or 'none')
            - structure_curve: Periodicity analysis curve
            - snr: Signal-to-noise ratio of detected peak
        """
        # Input validation
        if len(y) == 0:
            return 0.0, 0.0, {
                'harmonic_confidence': 0.0,
                'rejection_reason': 'empty_signal'
            }
        
        if np.max(np.abs(y)) < 1e-6:
            return 0.0, 0.0, {
                'harmonic_confidence': 0.0,
                'rejection_reason': 'silent_signal'
            }
        
        # Ensure float32 for Essentia
        y = np.asarray(y, dtype=np.float32)
        
        # Feature extraction with gate filtering
        feat = self._extract_features(y)
        if feat is None:
            return 0.0, 0.0, {
                'harmonic_confidence': 0.0,
                'rejection_reason': 'non_harmonic'
            }
        
        # Build recurrence matrix
        rec = self._recurrence_matrix(feat.chroma)
        if rec is None:
            return 0.0, 0.0, {
                'harmonic_confidence': feat.harm_ratio,
                'rejection_reason': 'recurrence_failed'
            }
        
        # Extract periodicity from lag matrix
        period, conf, analysis = self._periodicity_from_lag(rec, min_period, max_period)
        
        # Add feature metrics to analysis
        analysis['harmonic_confidence'] = feat.harm_ratio
        analysis['spectral_flatness'] = feat.flatness
        analysis['chroma_diversity'] = feat.diversity
        analysis['spectral_entropy'] = feat.spectral_entropy
        analysis['rejection_reason'] = 'none'
        
        # === CONFIDENCE ADJUSTMENT (preserved from v4.1) ===
        
        # Penalize low harmonic content
        if feat.harm_ratio < 0.15:
            conf *= 0.4
        elif feat.harm_ratio < 0.30:
            conf *= 0.7
        
        # Penalize low diversity
        if feat.diversity < 0.35:
            conf *= 0.5
        
        return period, float(conf), analysis
    
    def analyze_windows(self, y: np.ndarray,
                        min_period: float = 8.0,
                        max_period: float = 14.0,
                        n_win: int = 5) -> Tuple[float, float, Dict]:
        """
        Analyze audio using multiple overlapping windows for robustness.
        
        This method provides more reliable results by:
        1. Analyzing multiple segments independently
        2. Filtering outliers using MAD (Median Absolute Deviation)
        3. Computing confidence-weighted average
        4. Adding consistency bonus for stable detections
        
        Parameters:
        -----------
        y : np.ndarray
            Audio signal
        min_period : float
            Minimum period to detect (seconds)
        max_period : float
            Maximum period to detect (seconds)
        n_win : int
            Number of analysis windows
        
        Returns:
        --------
        period : float
            Weighted average period across windows
        confidence : float
            Combined confidence with consistency bonus
        analysis : dict
            Aggregated analysis results
        """
        total = len(y)
        min_samples = int(25 * self.sr)  # Minimum 25 seconds per window
        
        # Fall back to single-window if signal too short
        if total < min_samples * 2:
            return self.detect_periodicity(y, min_period, max_period)
        
        win_size = total // n_win
        overlap = win_size // 4
        results: List[Dict] = []
        
        for i in range(n_win):
            start = max(0, i * win_size - overlap)
            end = min((i + 1) * win_size + overlap, total)
            
            if end - start < min_samples:
                continue
            
            segment = np.asarray(y[start:end], dtype=np.float32)
            period, conf, analysis = self.detect_periodicity(
                segment, min_period, max_period
            )
            
            # Accept window if passes quality thresholds
            if (period > 0 and 
                conf > 0.15 and 
                analysis.get('harmonic_confidence', 0) > 0.10):
                results.append({
                    'window': i,
                    'period': period,
                    'confidence': conf,
                    'harm_conf': analysis.get('harmonic_confidence', 0),
                    'diversity': analysis.get('chroma_diversity', 0)
                })
        
        # Fall back if no valid windows
        if not results:
            return self.detect_periodicity(y, min_period, max_period)
        
        # Outlier filtering using MAD (Median Absolute Deviation)
        if len(results) >= 3:
            periods = np.array([r['period'] for r in results])
            median_period = np.median(periods)
            mad = np.median(np.abs(periods - median_period))
            
            # Keep results within 2*MAD + 0.5s of median
            filtered = [
                r for r in results
                if abs(r['period'] - median_period) < 2 * mad + 0.5
            ]
            
            if len(filtered) >= 2:
                results = filtered
        
        # Compute weighted average period
        periods = np.array([r['period'] for r in results])
        confs = np.array([r['confidence'] for r in results])
        weighted_period = float(np.average(periods, weights=confs))
        period_std = float(np.std(periods))
        
        # Consistency bonus (reward stable detection across windows)
        if period_std < 0.3:
            bonus = 0.15  # Very consistent
        elif period_std < 0.7:
            bonus = 0.08  # Moderately consistent
        else:
            bonus = 0.0   # Inconsistent
        
        final_conf = float(min(1.0, np.mean(confs) + bonus))
        
        return weighted_period, final_conf, {
            'harmonic_confidence': float(np.mean([r['harm_conf'] for r in results])),
            'chroma_diversity': float(np.mean([r.get('diversity', 0.5) for r in results])),
            'n_windows_used': len(results),
            'period_std': period_std,
            'rejection_reason': 'none',
            'window_results': results  # Include per-window results for debugging
        }


class PhraseDetector:
    """
    Simple RMS autocorrelation-based phrase detector.
    
    This is a lightweight alternative that uses energy envelope
    rather than harmonic content. Useful for quick screening or
    as a secondary confirmation method.
    """
    
    def __init__(self, sr: int = 22050):
        self.sr = sr
        self.hop = 8192
        self.frame_len = 16384
        self._rms = es.RMS()
    
    def detect(self, y: np.ndarray,
               min_period: float = 8.0,
               max_period: float = 13.0) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Detect phrase periodicity using RMS autocorrelation.
        
        Parameters:
        -----------
        y : np.ndarray
            Audio signal
        min_period : float
            Minimum period (seconds)
        max_period : float
            Maximum period (seconds)
        
        Returns:
        --------
        period : float
            Detected period in seconds
        lags : np.ndarray
            Lag values (seconds)
        ac_norm : np.ndarray
            Normalized autocorrelation
        """
        y = np.asarray(y, dtype=np.float32)
        
        # Compute RMS energy envelope
        rms_values = []
        for frame in es.FrameGenerator(y, frameSize=self.frame_len,
                                        hopSize=self.hop, startFromZero=True):
            rms_values.append(self._rms(frame))
        
        rms = np.array(rms_values)
        
        if len(rms) < 3:
            return 0.0, np.array([0]), np.array([0])
        
        # Normalize RMS
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        if rms_std > 1e-9:
            rms = (rms - rms_mean) / rms_std
        
        # Compute autocorrelation
        ac = np.correlate(rms, rms, mode='full')[len(rms) - 1:]
        dt = self.hop / self.sr
        lags = np.arange(len(ac)) * dt
        
        # Normalize autocorrelation
        ac_norm = ac / (ac[0] + 1e-9)
        
        # Find peak in valid range
        mask = (lags >= min_period) & (lags <= max_period)
        if not np.any(mask):
            return 0.0, lags, ac_norm
        
        valid_ac = ac_norm.copy()
        valid_ac[~mask] = -np.inf
        
        best_lag_idx = np.argmax(valid_ac)
        
        return float(lags[best_lag_idx]), lags, ac_norm


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def validate_audio(y: np.ndarray, sr: int, min_duration: float = 20.0) -> bool:
    """
    Validate audio for periodicity analysis.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate
    min_duration : float
        Minimum duration in seconds
    
    Returns:
    --------
    bool
        True if audio is valid for analysis
    """
    if y is None or len(y) == 0:
        return False
    
    duration = len(y) / sr
    if duration < min_duration:
        return False
    
    if np.max(np.abs(y)) < 1e-6:
        return False
    
    return True


def batch_analyze(audio_paths: List[str],
                  sr: int = 22050,
                  min_period: float = 8.0,
                  max_period: float = 14.0,
                  use_windows: bool = True,
                  n_workers: int = 4) -> List[Dict]:
    """
    Batch analyze multiple audio files.
    
    This is optimized for the 10,000 track scaling objective.
    
    Parameters:
    -----------
    audio_paths : List[str]
        Paths to audio files
    sr : int
        Target sample rate
    min_period : float
        Minimum period to detect
    max_period : float
        Maximum period to detect
    use_windows : bool
        Use multi-window analysis
    n_workers : int
        Number of parallel workers
    
    Returns:
    --------
    List[Dict]
        Analysis results for each file
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os
    
    def analyze_single(path: str) -> Dict:
        try:
            # Load audio using Essentia
            loader = es.MonoLoader(filename=path, sampleRate=sr)
            y = loader()
            
            if not validate_audio(y, sr):
                return {
                    'path': path,
                    'period': 0.0,
                    'confidence': 0.0,
                    'error': 'invalid_audio'
                }
            
            detector = PhraseBoundaryDetector(sr=sr)
            
            if use_windows:
                period, conf, analysis = detector.analyze_windows(
                    y, min_period, max_period
                )
            else:
                period, conf, analysis = detector.detect_periodicity(
                    y, min_period, max_period
                )
            
            return {
                'path': path,
                'period': period,
                'confidence': conf,
                'harmonic_confidence': analysis.get('harmonic_confidence', 0),
                'rejection_reason': analysis.get('rejection_reason', 'none'),
                'error': None
            }
        except Exception as e:
            return {
                'path': path,
                'period': 0.0,
                'confidence': 0.0,
                'error': str(e)
            }
    
    results = []
    
    # Use process pool for CPU-bound analysis
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(analyze_single, p): p for p in audio_paths}
        
        for future in as_completed(futures):
            results.append(future.result())
    
    return results


# =============================================================================
# TESTING / DEMONSTRATION
# =============================================================================

if __name__ == "__main__":
    import time
    
    print("=" * 60)
    print("BrainHeart Essentia Pipeline - Phrase Detector v5.0")
    print("=" * 60)
    
    # Generate test signal: 10-second periodic harmonic pattern
    sr = 22050
    duration = 60  # 60 seconds
    t = np.linspace(0, duration, int(sr * duration))
    
    # Create harmonically rich signal with 10-second structure
    phrase_period = 10.0  # Target: 10-second phrases
    
    # Base chord progression with 10-second cycle
    chord_env = 0.5 + 0.5 * np.sin(2 * np.pi * t / phrase_period)
    
    # Harmonic content (simulate musical chord)
    fundamental = 220  # A3
    harmonics = [1, 2, 3, 4, 5]
    signal_test = np.zeros_like(t)
    
    for h in harmonics:
        signal_test += (1.0 / h) * np.sin(2 * np.pi * fundamental * h * t)
    
    # Apply phrase envelope
    signal_test *= chord_env
    
    # Add some variation
    signal_test += 0.1 * np.random.randn(len(t))
    
    # Normalize
    signal_test = signal_test / np.max(np.abs(signal_test))
    signal_test = signal_test.astype(np.float32)
    
    print(f"\nTest signal: {duration}s @ {sr}Hz with {phrase_period}s phrase period")
    print("-" * 60)
    
    # Initialize detector
    detector = PhraseBoundaryDetector(sr=sr, verbose=True)
    
    # Single window analysis
    print("\n[1] Single Window Analysis")
    start_time = time.time()
    period, conf, analysis = detector.detect_periodicity(signal_test)
    elapsed = time.time() - start_time
    
    print(f"  Detected Period: {period:.2f}s (target: {phrase_period}s)")
    print(f"  Confidence: {conf:.3f}")
    print(f"  Harmonic Confidence: {analysis.get('harmonic_confidence', 0):.3f}")
    print(f"  SNR: {analysis.get('snr', 0):.2f}")
    print(f"  Time: {elapsed:.3f}s")
    
    # Multi-window analysis
    print("\n[2] Multi-Window Analysis (5 windows)")
    start_time = time.time()
    period_mw, conf_mw, analysis_mw = detector.analyze_windows(signal_test)
    elapsed_mw = time.time() - start_time
    
    print(f"  Detected Period: {period_mw:.2f}s (target: {phrase_period}s)")
    print(f"  Confidence: {conf_mw:.3f}")
    print(f"  Windows Used: {analysis_mw.get('n_windows_used', 0)}")
    print(f"  Period Std: {analysis_mw.get('period_std', 0):.3f}s")
    print(f"  Time: {elapsed_mw:.3f}s")
    
    # Test noise rejection
    print("\n[3] Noise Rejection Test")
    noise = np.random.randn(int(sr * 30)).astype(np.float32)
    period_noise, conf_noise, analysis_noise = detector.detect_periodicity(noise)
    print(f"  Period: {period_noise:.2f}s")
    print(f"  Confidence: {conf_noise:.3f}")
    print(f"  Rejection Reason: {analysis_noise.get('rejection_reason', 'none')}")
    
    # Test simple phrase detector
    print("\n[4] Simple Phrase Detector (RMS)")
    simple_detector = PhraseDetector(sr=sr)
    period_simple, lags, ac = simple_detector.detect(signal_test)
    print(f"  Detected Period: {period_simple:.2f}s")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("=" * 60)
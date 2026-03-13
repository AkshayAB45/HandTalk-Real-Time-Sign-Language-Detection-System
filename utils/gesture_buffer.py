"""
utils/gesture_buffer.py
========================
Temporal smoothing buffer that stabilizes predictions by
collecting the last N frames and returning the majority-vote
prediction only when confidence exceeds a threshold.
"""

from collections import deque, Counter


class GestureBuffer:
    """
    Smooths gesture predictions using a sliding window majority vote.

    Args:
        window_size (int): Number of frames to consider. Default: 10.
        min_confidence (float): Minimum model confidence to enqueue. Default: 0.65.
        majority_threshold (float): Fraction of window that must agree. Default: 0.6.
    """

    def __init__(self, window_size=10, min_confidence=0.65, majority_threshold=0.6):
        self.window_size       = window_size
        self.min_confidence    = min_confidence
        self.majority_threshold = majority_threshold
        self._buffer           = deque(maxlen=window_size)
        self._last_stable      = None
        self._stable_count     = 0

    def push(self, label, confidence):
        """
        Add a new prediction to the buffer.

        Args:
            label      (str):   Predicted class label.
            confidence (float): Model confidence (0–1).
        """
        if confidence >= self.min_confidence:
            self._buffer.append(label)

    def get_stable_prediction(self):
        """
        Returns (label, fraction) if the majority vote exceeds
        majority_threshold, otherwise returns (None, 0.0).
        """
        if len(self._buffer) < max(1, self.window_size // 2):
            return None, 0.0

        counts = Counter(self._buffer)
        top_label, top_count = counts.most_common(1)[0]
        fraction = top_count / len(self._buffer)

        if fraction >= self.majority_threshold:
            return top_label, fraction
        return None, 0.0

    def clear(self):
        """Reset the buffer."""
        self._buffer.clear()
        self._last_stable = None
        self._stable_count = 0

    @property
    def is_full(self):
        return len(self._buffer) == self.window_size

    @property
    def fill_ratio(self):
        return len(self._buffer) / self.window_size

    def __len__(self):
        return len(self._buffer)


class SentenceBuilder:
    """
    Builds a translated sentence word by word.
    Prevents duplicate consecutive words and enforces
    a minimum gap between same-word detections.

    Args:
        max_words (int): Rolling window of words to display. Default: 15.
        cooldown_frames (int): Frames to wait before same word can be added again.
    """

    def __init__(self, max_words=15, cooldown_frames=20):
        self.max_words       = max_words
        self.cooldown_frames = cooldown_frames
        self._words          = deque(maxlen=max_words)
        self._last_word      = None
        self._frames_since   = 0

    def try_add(self, word):
        """
        Attempt to add a word to the sentence.
        Returns True if the word was added, False if blocked by cooldown.
        """
        self._frames_since += 1

        if word == self._last_word and self._frames_since < self.cooldown_frames:
            return False  # Too soon, same word

        self._words.append(word)
        self._last_word    = word
        self._frames_since = 0
        return True

    def get_sentence(self):
        """Returns the current sentence as a string."""
        return ' '.join(self._words)

    def clear(self):
        """Wipe the sentence."""
        self._words.clear()
        self._last_word  = None
        self._frames_since = 0

    def backspace(self):
        """Remove the last word."""
        if self._words:
            self._words.pop()
            self._last_word = self._words[-1] if self._words else None

    def __len__(self):
        return len(self._words)

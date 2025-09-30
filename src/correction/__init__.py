"""
Correction package initialization.
"""
from .gemini_corrector import GeminiTextCorrector, CorrectionResult
from .smart_corrector import SmartTextCorrector

__all__ = ['GeminiTextCorrector', 'SmartTextCorrector', 'CorrectionResult']

__all__ = ['GeminiTextCorrector', 'CorrectionResult']
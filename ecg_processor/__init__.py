"""
ECG сигнал процессор - готовый пакет
"""

# Просто импортируем все функции из ваших файлов
from .core import (
    load_config,
    load_signal, 
    build_label_matrix,
    build_dataframe,
    process_file,
    main as process_main
)

from .visualization import plot_signal_with_labels

__version__ = '1.0.0'
__all__ = [
    'load_config',
    'load_signal',
    'build_label_matrix', 
    'build_dataframe',
    'process_file',
    'plot_signal_with_labels',
    'process_main'
]
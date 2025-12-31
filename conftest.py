# conftest.py
import warnings
import traceback

def pytest_configure(config):
    # mostra stacktrace completo per quel warning specifico
    def _showwarning(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if "numpy.ndarray size changed" in text:
            print("\n=== WARNING STACKTRACE (numpy.ndarray size changed) ===")
            traceback.print_stack()
            print("=== END STACKTRACE ===\n")
        return warnings._showwarning_orig(message, category, filename, lineno, file, line)

    if not hasattr(warnings, "_showwarning_orig"):
        warnings._showwarning_orig = warnings.showwarning
        warnings.showwarning = _showwarning

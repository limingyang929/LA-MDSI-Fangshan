"""Run summarize six settings from any working directory."""
from pathlib import Path
import os
import importlib
import sys

if __name__ == '__main__':
    sys.dont_write_bytecode = True
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    if '--help' in sys.argv and 'six_setting_summaries' in ('summarize_corrected_results','summarize_results','make_figures','six_setting_summaries','make_outputs','inspect_weak_signal'):
        print(__doc__)
        print('This stage takes no analysis options and requires its completed upstream stages.')
        print('See 04_Documentation/02_Run_guide.md for the required run order.')
        raise SystemExit(0)
    if '--smoke' in sys.argv:
        os.environ['LA_MDSI_SMOKE'] = '1'
    sys.path.insert(0, str(Path(__file__).resolve().parent / '00_Library'))
    target = importlib.import_module('six_setting_summaries')
    if hasattr(target, 'main'):
        target.main()

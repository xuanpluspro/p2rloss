import io
import os
import unittest
from unittest.mock import patch

from progress import progress_bar


class Terminal(io.StringIO):
    def isatty(self):
        return True


class ProgressTest(unittest.TestCase):
    def test_terminal_iteration_preserves_items(self):
        stream = Terminal()
        with patch('progress.sys.stderr', stream), patch.dict(os.environ, {'P2R_PROGRESS': '1'}):
            with progress_bar(range(3), desc='Train 1/50') as bar:
                self.assertFalse(bar.disable)
                self.assertEqual(list(bar), [0, 1, 2])
        self.assertIn('Train 1/50', stream.getvalue())
        self.assertFalse(stream.closed)

    def test_tee_uses_controlling_terminal(self):
        stream = Terminal()
        captured_logs = io.StringIO()
        with patch('progress.sys.stderr', captured_logs), patch('builtins.open', return_value=stream):
            with patch.dict(os.environ, {'P2R_PROGRESS': '1'}):
                with progress_bar([1], desc='Val 1/50') as bar:
                    self.assertFalse(bar.disable)
                    self.assertEqual(list(bar), [1])
        self.assertTrue(stream.closed)
        self.assertEqual(captured_logs.getvalue(), '')

    def test_noninteractive_and_manual_disable(self):
        with patch('progress.sys.stderr', io.StringIO()), patch('builtins.open', side_effect=OSError):
            with patch.dict(os.environ, {'P2R_PROGRESS': '1'}):
                with progress_bar([1, 2]) as bar:
                    self.assertTrue(bar.disable)
                    self.assertEqual(list(bar), [1, 2])
        with patch('progress.sys.stderr', Terminal()), patch.dict(os.environ, {'P2R_PROGRESS': '0'}):
            with progress_bar([3]) as bar:
                self.assertTrue(bar.disable)
                self.assertEqual(list(bar), [3])


if __name__ == '__main__':
    unittest.main()

import sys
import threading
import itertools


class Spinner(object):

    def __init__(self, disable=False, force=False, stream=sys.stdout, cycle=None):
        _cycle = cycle or ['-', '/', '|', '\\']
        self.spinner_cycle = itertools.cycle(_cycle)
        self.disable = disable
        self.force = force
        self.stream = stream
        self.stop_running = None
        self.spin_thread = None

    def start(self):
        if self.disable:
            return
        if self.stream.isatty() or self.force:
            self.stop_running = threading.Event()
            self.spin_thread = threading.Thread(target=self.init_spin)
            self.spin_thread.start()

    def stop(self):
        if self.spin_thread:
            self.stop_running.set()
            self.spin_thread.join()

    def init_spin(self):
        while not self.stop_running.is_set():
            content_to_stream = next(self.spinner_cycle)
            self.stream.write(content_to_stream)
            self.stream.flush()
            self.stop_running.wait(0.25)
            self.stream.write(''.join(['\b'] * len(content_to_stream)))
            self.stream.flush()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.disable:
            return False
        self.stop()
        return False

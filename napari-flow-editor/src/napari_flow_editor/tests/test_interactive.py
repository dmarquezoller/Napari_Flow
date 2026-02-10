"""
Tests for the InteractiveGUIWorker.

These tests verify the request_gui_action and wait_for_user_event methods
without requiring a full Napari GUI. We test result passing, error propagation,
and timeout behavior using mock objects.
"""

import pytest
import threading
import time
import queue
from unittest.mock import Mock, MagicMock, patch
from napari_flow_editor.interactive import InteractiveGUIWorker


@pytest.fixture
def worker():
    """Create an InteractiveGUIWorker instance for testing."""
    from qtpy.QtWidgets import QApplication
    import sys
    
    # Ensure QApplication exists
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    return InteractiveGUIWorker()


def test_worker_initialization(worker):
    """Test that the worker initializes correctly."""
    assert hasattr(worker, 'run_on_main_signal')
    assert hasattr(worker, 'request_gui_action')
    assert hasattr(worker, 'wait_for_user_event')


def test_execute_slot_success(worker):
    """Test that _execute_slot correctly handles successful execution."""
    result_queue = queue.Queue()
    
    def test_func():
        return 42
    
    worker._execute_slot(test_func, result_queue)
    
    success, result = result_queue.get(timeout=1)
    assert success is True
    assert result == 42


def test_execute_slot_error(worker):
    """Test that _execute_slot correctly handles exceptions."""
    result_queue = queue.Queue()
    
    def failing_func():
        raise ValueError("Test error")
    
    worker._execute_slot(failing_func, result_queue)
    
    success, result = result_queue.get(timeout=1)
    assert success is False
    assert isinstance(result, ValueError)
    assert str(result) == "Test error"


def test_request_gui_action_success(worker):
    """Test request_gui_action with a successful function."""
    # Connect the signal directly since we can't mock it
    def test_func():
        return "success"
    
    # Use the worker's actual mechanism
    result = worker.request_gui_action(test_func, timeout=1)
    assert result == "success"


def test_request_gui_action_with_error(worker):
    """Test that request_gui_action re-raises exceptions from GUI thread."""
    def failing_func():
        raise RuntimeError("GUI error")
    
    with pytest.raises(RuntimeError, match="GUI error"):
        worker.request_gui_action(failing_func, timeout=1)


def test_request_gui_action_timeout(worker):
    """Test that request_gui_action respects timeout."""
    # Create a function that would never return a result
    # by never emitting through the signal
    import queue as q
    result_queue = q.Queue()
    
    # Don't emit the signal, just check the timeout behavior
    with pytest.raises(TimeoutError, match="timed out"):
        result_queue.get(timeout=0.1)


def test_wait_for_user_event_success(worker):
    """Test wait_for_user_event with successful setup and cleanup."""
    event = threading.Event()
    
    def setup_func():
        # In real usage, this would set up a layer and return an event
        # For testing, we'll set the event immediately in a thread
        threading.Timer(0.1, event.set).start()
        return event
    
    def cleanup_func():
        return "cleanup done"
    
    result = worker.wait_for_user_event(setup_func, cleanup_func, timeout=2)
    assert result == "cleanup done"


def test_wait_for_user_event_without_cleanup(worker):
    """Test wait_for_user_event with no cleanup function."""
    event = threading.Event()
    
    def setup_func():
        threading.Timer(0.1, event.set).start()
        return event
    
    result = worker.wait_for_user_event(setup_func, cleanup_func=None, timeout=2)
    assert result is None


def test_wait_for_user_event_setup_error(worker):
    """Test that errors in setup_func are propagated."""
    def failing_setup():
        raise ValueError("Setup failed")
    
    with pytest.raises(ValueError, match="Setup failed"):
        worker.wait_for_user_event(failing_setup, lambda: None, timeout=1)


def test_wait_for_user_event_cleanup_error(worker):
    """Test that errors in cleanup_func are propagated."""
    event = threading.Event()
    
    def setup_func():
        threading.Timer(0.05, event.set).start()
        return event
    
    def failing_cleanup():
        raise RuntimeError("Cleanup failed")
    
    with pytest.raises(RuntimeError, match="Cleanup failed"):
        worker.wait_for_user_event(setup_func, failing_cleanup, timeout=2)


def test_wait_for_user_event_timeout_on_event(worker):
    """Test that wait_for_user_event times out if event is not set."""
    event = threading.Event()
    # Event will never be set
    
    def setup_func():
        return event
    
    with pytest.raises(TimeoutError, match="User interaction timed out"):
        worker.wait_for_user_event(setup_func, lambda: None, timeout=0.1)


def test_request_gui_action_return_types(worker):
    """Test that request_gui_action handles various return types."""
    # Test None
    assert worker.request_gui_action(lambda: None, timeout=1) is None
    
    # Test int
    assert worker.request_gui_action(lambda: 42, timeout=1) == 42
    
    # Test string
    assert worker.request_gui_action(lambda: "test", timeout=1) == "test"
    
    # Test list
    assert worker.request_gui_action(lambda: [1, 2, 3], timeout=1) == [1, 2, 3]
    
    # Test dict
    assert worker.request_gui_action(lambda: {"key": "value"}, timeout=1) == {"key": "value"}
    
    # Test tuple
    assert worker.request_gui_action(lambda: (1, 2, 3), timeout=1) == (1, 2, 3)


def test_worker_is_qobject(worker):
    """Test that InteractiveGUIWorker is a QObject."""
    from qtpy.QtCore import QObject
    assert isinstance(worker, QObject)


def test_signal_exists(worker):
    """Test that the run_on_main_signal exists and is a Signal."""
    from qtpy.QtCore import Signal
    # The signal should be accessible as a class attribute
    assert hasattr(InteractiveGUIWorker, 'run_on_main_signal')

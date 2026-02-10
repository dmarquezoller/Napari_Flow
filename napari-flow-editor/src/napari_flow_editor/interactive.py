"""
Generic interactive GUI worker for nodes that need to pause execution
and request user input from Napari's GUI thread.

This module provides a reusable pattern for any node that needs to:
- Emit a signal to run a function on the main (GUI) thread
- Block the worker thread until the user finishes interacting
- Pass results back from the main thread to the worker thread
- Handle errors and optional timeouts
"""

import queue
import threading
from qtpy.QtCore import QObject, Signal


class InteractiveGUIWorker(QObject):
    """
    A generic worker that marshals function calls to the GUI thread.
    
    This class handles the threading/signal boilerplate for interactive nodes,
    allowing any node to pause pipeline execution, request user input from 
    Napari's GUI, and resume with the results.
    """
    
    run_on_main_signal = Signal(object, object)
    
    def __init__(self):
        """Initialize the worker and connect the signal to the execution slot."""
        super().__init__()
        self.run_on_main_signal.connect(self._execute_slot)
    
    def _execute_slot(self, func, result_queue):
        """
        Execute a function on the main thread and put the result in the queue.
        
        Args:
            func: Callable to execute on the main thread
            result_queue: Queue to put (success, result) tuple
        """
        try:
            result = func()
            result_queue.put((True, result))
        except Exception as e:
            result_queue.put((False, e))
    
    def request_gui_action(self, gui_func, timeout=None):
        """
        Request a function to be executed on the main GUI thread.
        
        Blocks the worker thread until the GUI function completes, then returns
        the result. If the GUI function raises an exception, it is re-raised 
        on the worker thread.
        
        Args:
            gui_func: Callable that will be executed on the main thread.
                     Should return the desired result.
            timeout: Optional timeout in seconds. If None, waits indefinitely.
        
        Returns:
            The return value of gui_func
            
        Raises:
            The exception raised by gui_func, if any
            queue.Empty: If timeout expires before result is available
        """
        result_queue = queue.Queue()
        self.run_on_main_signal.emit(gui_func, result_queue)
        
        # Wait for result with optional timeout
        try:
            success, result = result_queue.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(f"GUI action timed out after {timeout} seconds")
        
        # Re-raise exception from GUI thread if one occurred
        if not success:
            raise result
        
        return result
    
    def wait_for_user_event(self, setup_func, cleanup_func=None, timeout=None):
        """
        Convenience method for the common pattern of:
        1. Set up a Napari layer for user interaction
        2. Wait for the user to complete their interaction
        3. Collect the result and clean up
        
        This is a high-level wrapper around request_gui_action that handles
        the setup-wait-cleanup pattern common in interactive nodes.
        
        Args:
            setup_func: Callable that sets up the interaction on the main thread.
                       Should return a threading.Event that will be set when 
                       the user completes their interaction.
            cleanup_func: Optional callable to clean up and retrieve final data
                         on the main thread. Called after the event is set.
                         If None, no cleanup is performed.
            timeout: Optional timeout in seconds for both setup and cleanup.
        
        Returns:
            The return value of cleanup_func (if provided), or None
            
        Raises:
            Any exception raised by setup_func or cleanup_func
            TimeoutError: If timeout expires
        """
        # Step 1: Set up the interaction and get an Event to wait on
        event = self.request_gui_action(setup_func, timeout=timeout)
        
        # Step 2: Wait for the user to complete their interaction
        # The event should be set by a callback in the GUI thread
        if timeout is not None:
            # Wait with timeout
            if not event.wait(timeout=timeout):
                raise TimeoutError(f"User interaction timed out after {timeout} seconds")
        else:
            # Wait indefinitely
            event.wait()
        
        # Step 3: Clean up and retrieve the result
        if cleanup_func is not None:
            return self.request_gui_action(cleanup_func, timeout=timeout)
        
        return None


# Global instance for use across nodes
# Nodes can import this directly: from napari_flow_editor.interactive import interactive_worker
interactive_worker = InteractiveGUIWorker()

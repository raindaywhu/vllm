# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import threading
import queue
from queue import Queue
from typing import Callable, Optional, Any

from vllm.logger import init_logger

logger = init_logger(__name__)

class EPLBThread:
    """
    Encapsulates lifecycle management for asynchronous expert
    rearrangement threads
    """

    def __init__(self, target_func: Callable, num_wait_worker_iterations: int):
        logger.error("EPLBThread __init__")
        """
        Initialize asynchronous thread manager

        Args:
            target_func: Target function to execute in asynchronous thread
                (e.g., rebalance_experts)
            num_wait_worker_iterations: Number of steps to wait before
                checking results
        """
        self.target_func = target_func
        self._num_wait_worker_iterations = num_wait_worker_iterations

        # Thread management related
        self._thread: Optional[threading.Thread] = None
        self._input_queue: Optional[Queue] = None
        self._result_queue: Optional[Queue] = None
        self._exception_queue: Optional[Queue] = None
        self._step_counter = 0
        self._result: Optional[tuple] = None
        self._args: Optional[tuple] = None
        self._is_running = False
        self._has_pending_task = False
        self._is_post_processing = False
        self._stop_event = threading.Event()

        # Save parameters needed for post-processing
        self._post_process_args: Optional[dict[str, Any]] = None

        # Initialize thread and queues
        self._initialize_thread()

    def _initialize_thread(self) -> None:
        """Initialize the background thread and queues"""
        logger.error("EPLBThread _initialize_thread")
        try:
            # Initialize queues
            self._input_queue = Queue()
            self._result_queue = Queue()
            self._exception_queue = Queue()
            self._stop_event.clear()

            # Start the thread
            self._thread = threading.Thread(
                target=self._worker_loop,
                name="EPLBThread",
                args=(self._input_queue, self._result_queue, 
                      self._exception_queue, self._stop_event)
            )
            self._thread.daemon = True  # Set as daemon thread
            self._thread.start()
            self._is_running = True
            logger.debug("EPLB background thread started")

        except Exception as e:
            logger.error("Failed to start EPLB background thread: {}", str(e))
            self.cleanup()
            raise

    def _worker_loop(self, input_queue: Queue, output_queue: Queue,
                     exception_queue: Queue, stop_event: threading.Event) -> None:
        logger.error("EPLBThread _worker_loop")
        """Thread worker loop that processes tasks continuously"""
        try:
            while not stop_event.is_set():
                # Get arguments from input queue
                try:
                    args = input_queue.get(timeout=0.1)
                    if args is None:  # Sentinel value to stop the thread
                        break

                    # Execute target function
                    result = self.target_func(*args)
                    output_queue.put(result)
                except queue.Empty:
                    # Timeout, check if we should continue
                    continue
                except Exception as e:
                    output_queue.put(None)
                    if hasattr(e, "add_note"):
                        import traceback
                        e.add_note(traceback.format_exc())
                    exception_queue.put(e)
                    logger.exception("Task execution failed in worker thread")

        except Exception as e:
            exception_queue.put(e)
            logger.exception("Worker thread encountered fatal error")
        finally:
            logger.debug("EPLB worker thread exiting")

    def submit_task(self, args: tuple,
                    post_process_args: dict[str, Any]) -> bool:
        logger.error("EPLBThread submit_task")
        """
        Submit a task to the asynchronous thread

        Args:
            args: Tuple of arguments to pass to the target function
            post_process_args: Parameters needed for subsequent
                processing (e.g., model, ep_group)

        Returns:
            True if task submitted successfully, False otherwise
        """
        if not self._is_running:
            logger.error("Cannot submit task: thread is not running")
            return False

        if self._has_pending_task:
            logger.warning("Cannot submit task: already has a pending task")
            return False

        try:
            # Put arguments to the input queue
            self._input_queue.put(args)
            self._args = args
            self._post_process_args = post_process_args
            self._has_pending_task = True
            self._step_counter = 0
            self._result = None
            return True

        except Exception as e:
            logger.error("Failed to submit task to asynchronous thread: {}",
                         str(e))
            return False

    def step(self) -> bool:
        logger.error("EPLBThread step")
        """
        Increment step counter and check if results need processing

        Returns:
            Whether results have been processed
        """
        if not self._is_running or not self._has_pending_task:
            return False

        self._step_counter += 1

        # Check for exceptions first
        if self._exception_queue and not self._exception_queue.empty():
            error_msg = self._exception_queue.get()
            self._has_pending_task = False
            raise RuntimeError("Asynchronous thread failed: {}", error_msg)

        # Check if processing conditions are met
        if self._should_process():
            self._fetch_result()
            self._has_pending_task = False
            return True

        return False

    def _should_process(self) -> bool:
        logger.error("EPLBThread _should_process")
        """Determine if results need processing"""
        if not self._thread or not self._result_queue:
            return True

        return (self._step_counter >= self._num_wait_worker_iterations
                or not self._thread.is_alive()
                or not self._result_queue.empty())

    def _fetch_result(self) -> None:
        logger.error("EPLBThread _fetch_result")
        """Retrieve thread results"""
        if self._result_queue and not self._result_queue.empty():
            self._result = self._result_queue.get()
        else:
            self._result = None
            logger.warning(
                "Asynchronous thread completed but no result was returned")

    def cleanup(self) -> None:
        logger.error("EPLBThread cleanup")
        """Clean up thread resources"""
        # Clear stop event first
        self._stop_event.clear()
        
        # Set stop event to signal thread to exit
        self._stop_event.set()

        # Send sentinel value to stop the thread
        if self._input_queue:
            try:
                self._input_queue.put(None)
            except:
                pass

        if self._thread and self._thread.is_alive():
            # Wait for thread to finish with timeout
            self._thread.join(timeout=1.0)
            self._thread = None

        self._input_queue = None
        self._result_queue = None
        self._exception_queue = None
        
        self._is_running = False
        self._has_pending_task = False

    @property
    def is_running(self) -> bool:
        # logger.error("EPLBThread is_running")
        """Return whether the thread is running"""
        return self._is_running

    @property
    def has_pending_task(self) -> bool:
        # logger.error("EPLBThread has_pending_task")
        """Return whether there is a pending task"""
        return self._has_pending_task

    @property
    def is_post_processing(self) -> bool:
        return self._is_post_processing

    @is_post_processing.setter
    def is_post_processing(self, value: bool):
        self._is_post_processing = value

    @property
    def result(self) -> Optional[tuple]:
        logger.error("EPLBThread result")
        """Return processing results"""
        return self._result

    @property
    def post_process_args(self) -> Optional[dict[str, Any]]:
        logger.error("EPLBThread post_process_args")
        """Return post-processing arguments"""
        return self._post_process_args

    def __del__(self):
        logger.error("EPLBThread __del__")
        """Ensure resource cleanup when object is destroyed"""
        self.cleanup()
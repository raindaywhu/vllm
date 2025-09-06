import pytest
import time

from vllm.distributed.eplb import EPLBThread


def dummy_target_function(arg1, arg2):
    """Simulate target function for testing"""
    return arg1 + arg2, arg1 * arg2


def test_eplb_thread_initialization():
    """Test EPLBThread initialization"""
    thread = EPLBThread(
        target_func=dummy_target_function,
        num_wait_worker_iterations=5
    )

    assert thread.target_func == dummy_target_function
    assert thread._num_wait_worker_iterations == 5
    assert thread._is_running is True
    assert thread._has_pending_task is False
    assert thread._input_queue is not None
    assert thread._result_queue is not None
    assert thread._exception_queue is not None
    assert thread._thread is not None
    assert thread._thread.is_alive() is True

    # Clean up
    thread.cleanup()


def test_eplb_thread_submit_task():
    """Test task submission functionality"""
    thread = EPLBThread(
        target_func=dummy_target_function,
        num_wait_worker_iterations=3
    )

    # Submit task
    success = thread.submit_task(
        args=(10, 5),
        post_process_args={"test": "data"}
    )

    assert success is True
    assert thread._has_pending_task is True
    assert thread._args == (10, 5)
    assert thread._post_process_args == {"test": "data"}

    # Clean up
    thread.cleanup()


def test_eplb_thread_submit_task_when_busy():
    """Test submitting new task when existing task is pending"""
    thread = EPLBThread(
        target_func=dummy_target_function,
        num_wait_worker_iterations=3
    )

    # Submit first task
    success1 = thread.submit_task(
        args=(10, 5),
        post_process_args={"test": "data1"}
    )

    # Attempt to submit second task
    success2 = thread.submit_task(
        args=(20, 10),
        post_process_args={"test": "data2"}
    )

    assert success1 is True
    assert success2 is False  # Should fail because task is pending

    # Clean up
    thread.cleanup()


def test_eplb_thread_step_without_pending_task():
    """Test step method without pending task"""
    thread = EPLBThread(
        target_func=dummy_target_function,
        num_wait_worker_iterations=3
    )

    # Call step without submitting task
    result = thread.step()

    assert result is False  # Should return False because no pending task

    # Clean up
    thread.cleanup()


def test_eplb_thread_step_with_pending_task():
    """Test step method with pending task"""
    thread = EPLBThread(
        target_func=dummy_target_function,
        num_wait_worker_iterations=1  # Only wait for one step
    )

    # Submit task
    thread.submit_task(
        args=(10, 5),
        post_process_args={"test": "data"}
    )

    # Wait for task to complete
    time.sleep(0.5)  # Give thread time to execute

    # Call step to check result
    result = thread.step()

    assert result is True  # Should return True, indicating result threaded
    assert thread.result == (15, 50)  # 10+5=15, 10 * 5=50
    assert thread._has_pending_task is False  # Task completed

    # Clean up
    thread.cleanup()


def test_eplb_thread_step_before_completion():
    """Test calling step method before task completion"""
    thread = EPLBThread(
        target_func=lambda x: time.sleep(0.2),  # Function that takes time
        num_wait_worker_iterations=10  # Requires multiple wait steps
    )

    # Submit task
    thread.submit_task(
        args=(0.2,),  # Sleep for 0.2 seconds
        post_process_args={"test": "data"}
    )

    # Call step immediately (task should not be completed yet)
    result = thread.step()

    assert result is False  # Should return False because task not completed
    assert thread._has_pending_task is True  # Task still in progress

    # Clean up
    thread.cleanup()


def test_eplb_thread_exception_handling():
    """Test exception handling"""

    def failing_function():
        raise ValueError("Test exception")

    thread = EPLBThread(
        target_func=failing_function,
        num_wait_worker_iterations=3
    )

    # Submit task that will fail
    thread.submit_task(
        args=(),
        post_process_args={"test": "data"}
    )

    # Wait for task to complete
    time.sleep(0.5)

    # Calling step should raise exception
    with pytest.raises(RuntimeError, match="Asynchronous thread failed"):
        thread.step()

    # Clean up
    thread.cleanup()


def test_eplb_thread_cleanup():
    """Test cleanup functionality"""
    thread = EPLBThread(
        target_func=dummy_target_function,
        num_wait_worker_iterations=3
    )

    # Ensure thread is running
    assert thread._is_running is True
    assert thread._thread.is_alive() is True

    # Perform cleanup
    thread.cleanup()

    # Check state
    assert thread._is_running is False
    assert thread._thread is None
    assert thread._input_queue is None
    assert thread._result_queue is None
    assert thread._exception_queue is None


if __name__ == "__main__":
    pytest.main([__file__])
import os
import signal
import subprocess
import sys
import unittest.mock
from textwrap import dedent

import pytest

from ray._private.test_utils import (
    get_error_message,
    run_string_as_driver,
    wait_for_condition,
)
from ray.raylet import AgentManager


def test_agent_manager_fate_sharing_mocked(monkeypatch):
    """
    Test agent manager behavior using a mocked process.
    This test validates the two outcomes of the race condition:
    1. AgentManager's wait() succeeds and gets the correct non-zero exit code.
    2. AgentManager's wait() fails with ECHILD because the process was
       already reaped.
    """
    raylet_stopper = unittest.mock.Mock()
    # The agent command is not actually run because we mock process_.Wait()
    agent_command = ["echo", "mock agent"]
    process_mock = unittest.mock.Mock()

    monkeypatch.setattr(subprocess, "Popen", lambda *args, **kwargs: process_mock)

    # Scenario 1: waitpid() returns a non-zero exit code (SIGKILL)
    process_mock.wait.return_value = 137  # 128 + 9 (SIGKILL)
    agent_manager = AgentManager(
        "mock_agent",
        raylet_stopper=raylet_stopper,
        agent_command=agent_command,
        fate_share=True,
    )
    agent_manager.start()
    agent_manager.join()
    raylet_stopper.assert_called_once_with(137)

    # Reset for Scenario 2
    raylet_stopper.reset_mock()
    process_mock.reset_mock()

    # Scenario 2: waitpid() fails with ECHILD (process already reaped)
    # This simulates the race condition where something else reaps the child.
    process_mock.wait.side_effect = subprocess.CalledProcessError(
        -1, "cmd", output="mocked error", stderr="ECHILD"
    )
    agent_manager_racy = AgentManager(
        "mock_agent",
        raylet_stopper=raylet_stopper,
        agent_command=agent_command,
        fate_share=True,
    )
    agent_manager_racy.start()
    agent_manager_racy.join()

    # In the racy case, we expect the stopper to be called with a generic
    # non-zero code because the actual exit code was lost.
    # The important part is that it is non-zero.
    raylet_stopper.assert_called_once()
    assert raylet_stopper.call_args[0][0] != 0 
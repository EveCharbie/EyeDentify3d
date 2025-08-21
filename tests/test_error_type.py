import unittest
import os
import io
import pytest
from unittest.mock import patch, mock_open
from eyedentify3d.error_type import ErrorType


def test_error_type_values():
    """Test that ErrorType enum has the expected values."""
    assert ErrorType.SKIP.value == "continues silently on errors"
    assert ErrorType.PRINT.value == "prints the error message to the console"
    assert ErrorType.FILE.value == "print the error message to a file"
    assert ErrorType.RAISE.value == "raises an exception on errors"


def test_error_type_skip():
    """Test that ErrorType.SKIP doesn't produce any output."""
    error_handler = ErrorType.SKIP
    # This should not produce any output
    error_handler("Test error message")
    # No assertion needed as we're testing that nothing happens


@patch("sys.stdout", new_callable=io.StringIO)
def test_error_type_print(mock_stdout):
    """Test that ErrorType.PRINT prints to console."""
    error_handler = ErrorType.PRINT
    error_handler("Test error message")
    assert mock_stdout.getvalue() == "Test error message"


@patch("builtins.open", new_callable=mock_open)
def test_error_type_file(mock_file):
    """Test that ErrorType.FILE writes to a file."""
    error_handler = ErrorType.FILE
    error_handler("Test error message")

    # Check that open was called with the right arguments
    mock_file.assert_called_once_with("bad_data_files.txt", "a")
    # Check that write was called with the right arguments
    mock_file().write.assert_called_once_with("Test error message\n")


def test_error_type_raise():
    """Test that ErrorType.RAISE raises a RuntimeError."""
    error_handler = ErrorType.RAISE
    with pytest.raises(RuntimeError, match="Test error message"):
        error_handler("Test error message")


def test_error_type_unknown():
    """Test that an unknown error type raises a ValueError."""
    with pytest.raises(ValueError, match="'invalid' is not a valid ErrorType"):
        error_handler = ErrorType("invalid")


def test_error_type_integration():
    """Integration test for all error types."""
    # Test SKIP
    ErrorType.SKIP("This should be skipped")

    # Test PRINT
    with patch("sys.stdout", new_callable=io.StringIO) as mock_stdout:
        ErrorType.PRINT("This should be printed")
        assert "This should be printed" in mock_stdout.getvalue()

    # Test FILE
    test_filename = "bad_data_files.txt"
    # Remove the file if it exists
    if os.path.exists(test_filename):
        os.remove(test_filename)

    try:
        ErrorType.FILE("This should be written to file")
        # Check that the file exists and contains the message
        assert os.path.exists(test_filename)
        with open(test_filename, "r") as f:
            content = f.read()
        assert "This should be written to file" in content
    finally:
        # Clean up
        if os.path.exists(test_filename):
            os.remove(test_filename)

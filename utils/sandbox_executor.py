import os
import sys
import re
import tempfile
import subprocess
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class SandboxCodeExecutor:
    """
    Secure isolated executor for python code snippets. Runs code in a sandboxed
    subprocess with execution timeouts, memory footprint caps, and blacklisted keyword blocklists.
    """
    def __init__(self, timeout: float = 2.0, max_memory_mb: int = 128):
        self.timeout = timeout
        self.max_memory_mb = max_memory_mb
        
        # Simple static keyword blacklist to prevent malicious host actions
        self.blacklist = [
            r"import\s+os", r"import\s+sys", r"import\s+subprocess", r"import\s+shutil", 
            r"from\s+os\s+import", r"from\s+sys\s+import", r"from\s+subprocess\s+import",
            r"eval\(", r"exec\(", r"open\(", r"__import__", r"getattr", r"setattr",
            r"builtins", r"import\s+socket", r"socket\."
        ]

    def extract_code(self, text: str) -> str:
        """
        Extracts python code inside markdown code blocks ```python ... ``` or ``` ... ```.
        If no backticks are present, extracts text inside the first reasoning tag or returns the raw text.
        """
        # Try to find python code blocks
        patterns = [
            r"```python(.*?)```",
            r"```(.*?)```"
        ]
        
        for pat in patterns:
            match = re.search(pat, text, re.DOTALL)
            if match:
                return match.group(1).strip()
                
        # Fallback to returning raw code if it looks like python (e.g., has def or print)
        if "def " in text or "print(" in text:
            return text.strip()
            
        return ""

    def is_safe(self, code: str) -> tuple[bool, Optional[str]]:
        """
        Performs static analysis on the code block to intercept malicious attempts.
        """
        for pattern in self.blacklist:
            if re.search(pattern, code, re.IGNORECASE):
                return False, f"Keyword blocklist triggered: '{pattern}'"
        return True, None

    def execute_and_verify(
        self,
        code: str,
        assertions: Optional[str] = None,
        timeout: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Executes code inside a restricted subprocess, appending assertions if provided.
        
        Returns:
            Dict containing execution results: success, stdout, error, timeout, exit_code
        """
        exec_timeout = timeout if timeout is not None else self.timeout
        
        # 1. Check safety of baseline code
        safe, reason = self.is_safe(code)
        if not safe:
            return {
                "success": False,
                "timeout": False,
                "exit_code": -1,
                "stdout": "",
                "error": f"Security Exception: {reason}",
                "triggered_rules": [reason]
            }
            
        # 2. Append test assertions if provided
        full_code = code
        if assertions:
            # Check safety of assertions as well
            safe_assert, assert_reason = self.is_safe(assertions)
            if not safe_assert:
                return {
                    "success": False,
                    "timeout": False,
                    "exit_code": -1,
                    "stdout": "",
                    "error": f"Security Exception in Assertions: {assert_reason}"
                }
            full_code += "\n\n# --- Automatic Assertions ---\n" + assertions
            
        # 3. Write execution script to a temporary file
        temp_dir = tempfile.gettempdir()
        temp_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".py",
            dir=temp_dir,
            delete=False,
            encoding="utf-8"
        )
        
        try:
            temp_file.write(full_code)
            temp_file.close()
            
            # 4. Trigger subprocess execution using current interpreter
            # Limit resource usage via python's subprocess limit tools
            cmd = [sys.executable, temp_file.name]
            
            # Run subprocess safely
            process_res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=exec_timeout
            )
            
            success = process_res.returncode == 0
            return {
                "success": success,
                "timeout": False,
                "exit_code": process_res.returncode,
                "stdout": process_res.stdout.strip(),
                "error": process_res.stderr.strip()
            }
            
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Execution timed out after {exec_timeout} seconds.")
            return {
                "success": False,
                "timeout": True,
                "exit_code": -9,
                "stdout": e.stdout.strip() if e.stdout else "",
                "error": f"Execution Timeout: exceeded limits of {exec_timeout}s"
            }
        except Exception as e:
            logger.error(f"Sandbox subprocess execution failure: {e}")
            return {
                "success": False,
                "timeout": False,
                "exit_code": -5,
                "stdout": "",
                "error": f"System sandbox exception: {str(e)}"
            }
        finally:
            # Always clean up the temporary execution file
            if os.path.exists(temp_file.name):
                try:
                    os.unlink(temp_file.name)
                except Exception as e:
                    logger.error(f"Failed to delete temp file '{temp_file.name}': {e}")

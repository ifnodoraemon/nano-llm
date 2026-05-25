import unittest
from utils.sandbox_executor import SandboxCodeExecutor

class TestSandbox(unittest.TestCase):
    def setUp(self):
        self.executor = SandboxCodeExecutor(timeout=1.0)

    def test_extract_code(self):
        text = """
Some generic description.
```python
def multiply(a, b):
    return a * b
print(multiply(3, 4))
```
Footer text.
"""
        code = self.executor.extract_code(text)
        self.assertIn("def multiply", code)
        self.assertIn("print(multiply(3, 4))", code)

    def test_is_safe(self):
        # 1. Test safe code
        code_safe = "print('Hello world!')\n"
        safe, reason = self.executor.is_safe(code_safe)
        self.assertTrue(safe)
        
        # 2. Test blacklisted os import
        code_unsafe = "import os\nos.system('ls')\n"
        safe, reason = self.executor.is_safe(code_unsafe)
        self.assertFalse(safe)
        self.assertIn("Keyword blocklist triggered", reason)

    def test_execution_success(self):
        code = """
def add_nums(a, b):
    return a + b
print(add_nums(5, 7))
"""
        res = self.executor.execute_and_verify(code)
        self.assertTrue(res["success"])
        self.assertEqual(res["stdout"], "12")
        self.assertEqual(res["exit_code"], 0)

    def test_execution_syntax_error(self):
        code = """
def broken_syntax(a, b)
    return a + b
"""
        res = self.executor.execute_and_verify(code)
        self.assertFalse(res["success"])
        self.assertNotEqual(res["exit_code"], 0)
        self.assertIn("SyntaxError", res["error"])

    def test_execution_timeout(self):
        code = """
import time
while True:
    time.sleep(0.1)
"""
        # Modify blacklist temporarily to allow time import for timeout check
        # and trigger a safe loop
        self.executor.blacklist = [r"import\s+os", r"import\s+sys"]
        res = self.executor.execute_and_verify(code, timeout=0.5)
        self.assertFalse(res["success"])
        self.assertTrue(res["timeout"])
        self.assertIn("Execution Timeout", res["error"])

if __name__ == "__main__":
    unittest.main()

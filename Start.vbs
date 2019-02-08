Set oShell = CreateObject ("WScript.Shell") 
oShell.Run "cmd /k .\venv\Scripts\activate && python index.py", 0, false
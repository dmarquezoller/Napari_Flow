import os

print("\n--- PROJECT STRUCTURE ---")
for root, dirs, files in os.walk("."):
    # Ignore hidden folders like .git or __pycache__
    dirs[:] = [d for d in dirs if d not in [".git", "__pycache__", ".pytest_cache", ".vscode"]]
    
    level = root.count(os.sep)
    indent = ' ' * 4 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 4 * (level + 1)
    for f in files:
        if not f.endswith(".pyc") and f != ".DS_Store":
            print(f"{subindent}{f}")
print("-------------------------\n")
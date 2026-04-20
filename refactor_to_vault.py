import os

replacements = [
    ("memory_layer", "memory_vault"),
    ("Memory-Layer-AI", "Memory-Vault-AI"),
    ("Memory Layer AI", "Memory Vault AI"),
    ("Memory Layer", "Memory Vault"),
    ("memory-layer-ai", "memory-vault-ai"),
    ("memorylayer", "memoryvault"),
    ("memory-layer", "memory-vault"),
]

target_dirs = ["memory_layer", "tests", "docs", "scripts", "docker"]
root_files = ["pyproject.toml", "AGENTS.md", "README.md", "ROADMAP.md", "CHANGELOG.md", "CONTRIBUTING.md", "Dockerfile", "docker-compose.yml", "mkdocs.yml", ".env.example"]

# Process root files
for file in root_files:
    if os.path.exists(file):
        try:
            with open(file, "r", encoding="utf-8") as f:
                content = f.read()
            new_content = content
            for old, new in replacements:
                new_content = new_content.replace(old, new)
            if new_content != content:
                with open(file, "w", encoding="utf-8") as f:
                    f.write(new_content)
                print(f"Updated {file}")
        except Exception as e:
            print(f"Error reading {file}: {e}")

# Process target dirs
for d in target_dirs:
    if not os.path.exists(d): continue
    for root, dirs, files in os.walk(d):
        for file in files:
            if file.endswith((".py", ".md", ".json", ".txt", ".sh", ".yml", ".yaml")):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    new_content = content
                    for old, new in replacements:
                        new_content = new_content.replace(old, new)
                    if new_content != content:
                        with open(path, "w", encoding="utf-8") as f:
                            f.write(new_content)
                        print(f"Updated {path}")
                except Exception as e:
                    print(f"Error reading {path}: {e}")

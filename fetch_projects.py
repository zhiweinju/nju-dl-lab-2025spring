import re
import shutil
import subprocess
from pathlib import Path

# List of repository URLs with their titles
repos = [
    (
        "多模态谣言检测",
        "https://github.com/ShipingGe/NJUProject_MultimodalRumorDetection",
    ),
    ("参数高效微调（PEFT）", "https://github.com/Andre-Eads/NJUProject_PEFT"),
    ("强化学习", "https://github.com/fuyuchenIfyw/NJU_DL2025_project_RL.git"),
    (
        "检索增强生成（RAG）",
        "https://github.com/JinguoWang/NJU_2025spring_ragprojects.git",
    ),
    ("decode长度预测", "https://github.com/spliii/Generation_Length_Prediction"),
    ("KV cache缓存策略探索", "https://github.com/spliii/Caching-Strategy"),
    (
        "大模型训练性能Profiling和优化",
        "https://github.com/njuzyh/Profiling-optimization",
    ),
    ("大模型训练过程中的显存换出换入探索", "https://github.com/zb-nju/TRAINING_SWAP"),
    (
        "MoE Offloading：混合专家模型推理显存换入换出",
        "https://github.com/zzhbrr/NJUProject_MoE_Offloading",
    ),
    (
        "投机解码的draft策略分析",
        "https://github.com/zzhbrr/NJUProject_Speculative_Decoding_Draft_Strategy",
    ),
    ("干预LLM", "https://github.com/gjw185/NJU_steer"),
    ("智能工具调用代理系统", "https://github.com/umnooob/NJUProject_toolagent"),
    ("轻量级多Agent协作系统", "https://github.com/umnooob/NJUProject_multiagent"),
]


def clean_repo_name(url):
    """Extract clean name from repository URL"""
    return url.split("/")[-1].replace(".git", "")


def copy_images_and_update_paths(content, repo_path, target_dir, repo_name):
    """Copy images referenced in the README and update their paths"""
    # Create images directory
    images_dir = target_dir / "images" / repo_name
    images_dir.mkdir(parents=True, exist_ok=True)

    # Find all image references in markdown
    # This pattern matches both ![alt](path) and <img src="path" ...> formats
    img_patterns = [
        r"!\[(?:[^\]]*)\]\(([^)]+)\)",  # ![alt](path)
        r'<img[^>]+?src=["\'](.*?)["\']',  # <img src="path"
    ]

    new_content = content
    for pattern in img_patterns:
        matches = re.finditer(pattern, content)
        for match in matches:
            img_path = match.group(1)

            # Skip external URLs
            if img_path.startswith(("http://", "https://", "//")):
                continue

            # Convert the image path to absolute path relative to repo
            img_path = img_path.lstrip("/")
            src_img_path = repo_path / img_path

            if src_img_path.exists():
                # Copy image to new location
                dest_img_path = images_dir / src_img_path.name
                shutil.copy2(src_img_path, dest_img_path)

                # Update the path in the content
                rel_path = f"../images/{repo_name}/{src_img_path.name}"
                new_content = new_content.replace(img_path, rel_path)

    return new_content


def update_main_doc(projects_info):
    """Update the main document with local links"""
    main_doc = Path("docs/final/课程大作业.md")
    if not main_doc.exists():
        print("Warning: Main document not found!")
        return

    with open(main_doc, "r", encoding="utf-8") as f:
        content = f.read()

    # Update each project link
    for title, url in projects_info:
        repo_name = clean_repo_name(url)
        # Replace the GitHub URL with local path
        old_link = f"[{title}]({url})"
        new_link = f"[{title}](./projects/{repo_name}.md)"
        content = content.replace(old_link, new_link)

    # Write the updated content back
    with open(main_doc, "w", encoding="utf-8") as f:
        f.write(content)


def main():
    # Create projects directory if it doesn't exist
    projects_dir = Path("projects")
    projects_dir.mkdir(exist_ok=True)

    # Create docs directory if it doesn't exist
    docs_dir = Path("docs/final/projects")
    docs_dir.mkdir(parents=True, exist_ok=True)

    projects_info = []
    # Clone repositories and copy READMEs
    for title, repo_url in repos:
        repo_name = clean_repo_name(repo_url)
        print(f"\nProcessing {repo_name}...")

        # Clone repository
        repo_path = projects_dir / repo_name
        if not repo_path.exists():
            print(f"Cloning {repo_url}...")
            subprocess.run(["git", "clone", repo_url, str(repo_path)], check=True)
        else:
            print(f"Repository {repo_name} already exists, pulling latest changes...")
            subprocess.run(["git", "-C", str(repo_path), "pull"], check=True)

        # Copy README
        readme_path = repo_path / "README.md"
        if readme_path.exists():
            # Create target directory
            target_path = docs_dir / f"{repo_name}.md"
            print(f"Copying README to {target_path}")

            # Read README content
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Process images and update content
            content = copy_images_and_update_paths(
                content, repo_path, docs_dir, repo_name
            )

            # Write to target location with added header
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(f"# {title}\n\n")
                f.write(content)

            projects_info.append((title, repo_url))
        else:
            print(f"Warning: No README.md found in {repo_name}")

    # # Update the main document with local links
    # update_main_doc(projects_info)


if __name__ == "__main__":
    main()

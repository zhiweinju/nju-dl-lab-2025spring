## mkdocs使用
教程：https://www.mkdocs.org/user-guide/

1. 安装mkdocs-material
```bash
pip install mkdocs-material
```

2. 本地启动mkdocs
```bash
mkdocs serve
```

## 修改及提交内容

1. 修改docs文件夹下的内容，可以新建文件夹放置新实验
2. 修改mkdocs.yml文件，添加新实验的导航
```yaml
nav:
  - 主页: "index.md"
  - 实验一: 
    - 1.1 实验说明: "lab1/实验一介绍.md"
    - ...
  - 主tab名字:
    - 子tab名字: 相对docs文件夹的md文件相对路径
    - ...
```
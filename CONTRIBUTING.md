# 贡献指南

感谢您对《AI Agent 与大模型应用开发实战手册》的关注！欢迎提交 Issue 和 Pull Request。

---

## 🤝 如何贡献

### 1. 报告问题

如果您发现文档错误、代码 bug 或内容缺失，请：

1. 在 GitHub Issues 中搜索是否已有类似问题
2. 如果没有，创建新的 Issue 并选择合适的标签：
   - `bug` - 错误报告
   - `docs` - 文档问题
   - `enhancement` - 功能建议
   - `question` - 问题咨询

### 2. 提交代码修改

#### 分支管理

- `main` - 稳定版本
- `dev` - 开发版本（所有 PR 合并到此）

#### Pull Request 流程

1. **Fork** 本仓库
2. 从 `dev` 分支创建新分支：
   ```bash
   git checkout -b feature/your-feature-name
   # 或修复 bug：
   git checkout -b fix/description-of-bug
   ```
3. 进行修改并提交：
   ```bash
   git commit -m "feat: 添加新功能描述"
   # 或修复：
   git commit -m "fix: 修复 xxx 问题"
   ```
4. 推送分支：
   ```bash
   git push origin feature/your-feature-name
   ```
5. 在 GitHub 上创建 Pull Request 到 `dev` 分支

#### 提交信息规范

参考 [Conventional Commits](https://www.conventionalcommits.org/)：

| 类型 | 说明 |
|------|------|
| `feat:` | 新功能 |
| `fix:` | Bug 修复 |
| `docs:` | 文档修改 |
| `style:` | 代码格式（不影响功能） |
| `refactor:` | 重构（不影响功能） |
| `test:` | 测试相关 |
| `chore:` | 构建/工具相关 |

### 3. 贡献内容类型

#### 文档改进
- 修正错别字或语法错误
- 改进解释的清晰度
- 添加缺失的代码注释
- 完善缺失的章节

#### 代码贡献
- 修复代码 bug
- 优化性能
- 添加新功能
- 补充测试用例

#### 新增内容
- 新的动手实验项目
- 新章节内容
- 新的实战案例

---

## 📋 代码规范

### Python 代码

- 遵循 PEP 8 规范
- 使用有意义的变量和函数命名
- 添加必要的 docstring
- 确保代码可运行

### 文档格式

- Markdown 格式
- 中文标点符号
- 代码块标注语言
- 适当使用表格和列表

---

## ❓ 常见问题

**Q: 我的 PR 会被接受吗？**

A: 所有符合项目定位（AI Agent 实战内容）、质量达标的贡献都会被接受。大型改动建议先创建 Issue 讨论。

**Q: 可以添加广告或推广内容吗？**

A: 不可以。本项目不接受任何形式的广告、推广链接或商业宣传。

**Q: 如何联系维护者？**

A: 可以在 GitHub Issues 中留言，或通过项目主页的联系方式联系。

---

## 📜 许可证

参与本项目即表示您同意您的贡献将按照项目的[混合许可证](LICENSE)发布。

- 代码贡献采用 MIT License
- 文档贡献采用 CC BY-NC-SA 4.0

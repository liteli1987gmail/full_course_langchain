# 设置环境变量 http_proxy

假如可以正常访问Google，但是运行代码显示连接错误。最大可能性是没有设置环境变量 http_proxy
以下有三个方法解决这个问题。

- 代码：
import  os os.environ['HTTP_PROXY'] = 'http://127.0.0.1:端口号
os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:端口号

- Linux 和 macOS 设置环境变量 http_proxy
export http_proxy=http://127.0.0.1:端口号
export https_proxy=http://127.0.0.1:端口号

- window PowerShell
$env:http_proxy = "http://127.0.0.1:端口号"
$env:https_proxy= "http://127.0.0.1:端口号"


- 电脑的环境变量配置
在 Windows 中设置环境变量可以通过系统属性对话框来完成，这对所有新启动的进程都会生效。以下是设置环境变量的步骤：

1. **打开环境变量设置**:
   - 右键点击“此电脑”或“我的电脑”，选择“属性”。
   - 在打开的窗口中，点击“高级系统设置”。
   - 在“系统属性”对话框中，切换到“高级”选项卡。
   - 点击“环境变量”按钮。

2. **添加或修改环境变量**:
   - 在“环境变量”对话框中，你可以选择添加新的环境变量，或者编辑现有的环境变量。
   - 要设置 HTTP 代理，你可以在“系统变量”区域点击“新建”，然后输入变量名 `http_proxy` 和相应的值（例如 `http://127.0.0.1:端口号`）。
   - 如果你也需要设置 HTTPS 代理，可以同样方式添加 `https_proxy` 环境变量。

3. **应用更改**:
   - 完成设置后，点击“确定”保存更改。
   - 你可能需要重启计算机或重新启动相关应用程序，以使环境变量更改生效。

请注意，更改环境变量可能会影响所有使用这些变量的应用程序。在进行更改之前，请确保你了解这些设置的影响，并在必要时咨询 IT 专业人员。如果你只是想为特定的命令行会话设置代理，可以在命令行工具（如 PowerShell 或命令提示符）中临时设置环境变量，这样做不会影响系统级别的设置。
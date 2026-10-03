# 构建与使用

## Windows 构建

准备 Git、CMake 3.25 或更新版本、Ninja，以及支持 C++20 的 Visual Studio C++ 工具和 Windows SDK。Visual Studio 安装器中可选择“使用 C++ 的桌面开发”，并安装“适用于 Windows 的 C++ CMake 工具”。

在 **Visual Studio 的开发者 PowerShell（x64 编译环境）** 中执行。也可以从 x64 Native Tools Command Prompt 启动 PowerShell，以继承编译环境。下面的命令不依赖仓库中的内部脚本。

### 获取源码

```powershell
git clone --recurse-submodules https://github.com/cubeww/OpenSV.git
cd OpenSV
```

如果已经克隆仓库，或更新了主仓库代码，再执行：

```powershell
git submodule update --init --recursive
```

JUCE 位于 `third_party/JUCE`，版本由主仓库记录的子模块提交固定。

### 编译与启动

```powershell
cmake -S . -B build/Release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/Release --target OpenSV --parallel 4
& './build/Release/OpenSV_artefacts/Release/OpenSV.exe'
```

可在启动时直接打开工程：

```powershell
& './build/Release/OpenSV_artefacts/Release/OpenSV.exe' 'C:/Music/song.svp'
```

调试构建使用独立目录：

```powershell
cmake -S . -B build/Debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
cmake --build build/Debug --target OpenSV --parallel 4
& './build/Debug/OpenSV_artefacts/Debug/OpenSV.exe'
```

默认构建桌面编辑器及其合成库。macOS 尚未验证。

## Linux 构建

已在 Ubuntu 24.04.2（WSL、x86_64）上使用 GCC 13.3、CMake 3.28.3 和 Ninja 1.11 完成 Release 编译与链接。Linux 下的桌面交互、实际音频播放和声库合成运行尚未验证，其他发行版与 CPU 架构也未验证。

Ubuntu 24.04 的构建依赖可通过以下命令安装：

```bash
sudo apt update
sudo apt install build-essential cmake ninja-build pkg-config git \
    libasound2-dev libx11-dev libxext-dev libxrandr-dev libxinerama-dev \
    libxcursor-dev libxi-dev libfreetype-dev libfontconfig1-dev
```

获取源码并构建：

```bash
git clone --recurse-submodules https://github.com/cubeww/OpenSV.git
cd OpenSV
cmake -S . -B build/linux-release -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/linux-release --target OpenSV --parallel 4
./build/linux-release/OpenSV_artefacts/Release/OpenSV
```

已有仓库时，先在仓库根目录执行 `git submodule update --init --recursive`。Linux 与 Windows 使用不同构建目录，不可共用 CMake 缓存。运行编辑器需要可用的图形会话和音频设备；中文显示需要系统安装中文字体。

## 首次使用

声库和发音词典不随仓库提供，需要自行准备可用的 `voice.nofs` 和 `clf-data` 目录。

1. 新建工程或打开 `.svp` 文件。点击编曲区音轨左侧的声库名称，选择该歌手的 `voice.nofs`；每条发声轨分别配置。
2. 打开右侧“属性”面板，设置演唱语言和歌词词典目录 `clf-data`。已有声库信息的工程在本机找不到文件时，也需要重新选择路径。
3. 双击钢琴卷帘空白处添加音符，双击音符编辑歌词。也可在音符属性中填写显式音素，多个音素用空格分隔，例如日语 `r a`。
4. 按空格播放或暂停。后台合成未完成时会等待结果，编辑后自动更新受影响的部分；通过“视图”菜单可设置音频设备。
5. 使用 Ctrl+S 保存工程，通过“文件 → 导出歌声 WAV…”导出音频。

保存工程会记录声库及词典路径，不会把这些文件打包进工程；移动工程后需确保引用路径仍可访问。

## 歌词输入

选中音符后，右键选择“填入歌词…”或按 Ctrl+L，可按时间顺序批量填词。默认以空格、制表符或换行分隔；“按字符隔开”适合将汉字逐字分配，“循环填充”会重复歌词直到填满所选音符。未启用循环时，歌词不足则保留剩余音符，超出的歌词会忽略。整次填词可一次撤销。

歌词分配与读音转换是两个步骤。当前普通话支持词典收录的单字或拼音，日语支持词典中的罗马字键，粤语支持词典中的粤拼键；英语需填写显式音素，西班牙语词典支持尚不完整。普通话按一音符一音节输入，多音字可用拼音或显式音素指定读音。

显式音素优先于歌词，且须符合所选声库的语言音素表；全部音符都使用显式音素时，无需歌词词典。遇到未知歌词错误时，检查语言、词典目录和读音输入。

## 常用操作

| 操作 | 方法 |
| --- | --- |
| 添加音符 | 双击空白网格，或使用绘制工具拖动 |
| 移动 / 调整时值 | 拖动音符 / 拖动音符右边缘 |
| 多选 / 全选 | Ctrl 点击或框选 / Ctrl+A |
| 单个歌词编辑 | 双击音符；Enter 提交，Esc 取消 |
| 批量填词 | 选中音符后右键菜单或 Ctrl+L |
| 删除音符 | Delete |
| 钢琴键试听 | 点击卷帘左侧钢琴键 |
| 平移视野 | 中键拖动；滚轮上下，Shift + 滚轮左右 |
| 横向缩放 | Ctrl + 滚轮 |
| 播放 / 暂停 | 空格 |
| 打开 / 保存 / 另存 | Ctrl+O / Ctrl+S / Ctrl+Shift+S |
| 撤销 / 重做 | Ctrl+Z / Ctrl+Y |

点击区域标题可折叠，拖动区域间的分隔条可调整高度。播放游标越界后自动翻页，参数区跟随钢琴卷帘。

当前尚未支持续音符、共享组内部编辑和伴奏播放；声库兼容性及与原版的音质一致性仍在完善。

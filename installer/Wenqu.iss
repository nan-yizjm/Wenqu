#define MyAppName "Wenqu"
#define MyAppVersion "0.2.2"
#define MyAppPublisher "nan-yizjm"
#define MyAppExeName "Wenqu.exe"

[Setup]
AppId={{C36C1420-B87C-4E80-B2B2-89447061C8C7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Wenqu
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
OutputDir=..\dist-installer
OutputBaseFilename=Wenqu-Setup-{#MyAppVersion}-win-x64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\Wenqu\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 旧品牌的安装目录：升级安装（AppId 不变）不会自动清理它，留着会在磁盘上留一整套
; 旧程序。删的是**程序目录**，不是数据目录——资料、索引与会话在
; {localappdata}\ObsidianRAG（没有 Programs 这一层），两者互不相干。
[InstallDelete]
Type: filesandordirs; Name: "{localappdata}\Programs\ObsidianRAG"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{group}\用户指南"; Filename: "{app}\_internal\resources\docs\用户指南.md"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

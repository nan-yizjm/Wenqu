#define MyAppName "Obsidian RAG"
#define MyAppVersion "0.2.1"
#define MyAppPublisher "nan-yizjm"
#define MyAppExeName "ObsidianRAG.exe"

[Setup]
AppId={{C36C1420-B87C-4E80-B2B2-89447061C8C7}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\ObsidianRAG
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
OutputDir=..\dist-installer
OutputBaseFilename=ObsidianRAG-Setup-{#MyAppVersion}-win-x64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "..\dist\ObsidianRAG\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{group}\用户指南"; Filename: "{app}\_internal\resources\docs\用户指南.md"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

#ifndef MyAppName
  #define MyAppName "Wenqu"
#endif
#ifndef MyAppVersion
  #define MyAppVersion "0.2.3"
#endif
#define MyAppPublisher "nan-yizjm"
#ifndef MyAppExeName
  #define MyAppExeName "Wenqu.exe"
#endif
#ifndef BundleDir
  #define BundleDir "..\dist\Wenqu"
#endif
#ifndef MyAppId
  #define MyAppId "{{C36C1420-B87C-4E80-B2B2-89447061C8C7}"
#endif
#ifndef TestBuild
  #define TestBuild 0
#endif
#ifndef InstallerOutputDir
  #define InstallerOutputDir "..\dist-installer"
#endif

[Setup]
AppId={#MyAppId}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
DefaultDirName={localappdata}\Programs\Wenqu
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
OutputDir={#InstallerOutputDir}
OutputBaseFilename=Wenqu-Setup-{#MyAppVersion}-win-x64
Compression=lzma2
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}

[Files]
Source: "{#BundleDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

; 相同 AppId 升级沿用原安装位置。只移除本次安装目录内的旧入口，
; 不递归删除固定的旧品牌目录，也不触碰用户数据或其他安装。
[InstallDelete]
Type: files; Name: "{app}\ObsidianRAG.exe"
Type: files; Name: "{app}\ObsidianRAG.exe.manifest"

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon
Name: "{group}\用户指南"; Filename: "{app}\_internal\resources\docs\用户指南.md"

[Tasks]
Name: "desktopicon"; Description: "创建桌面快捷方式"; GroupDescription: "附加快捷方式："

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "启动 {#MyAppName}"; Flags: nowait postinstall skipifsilent

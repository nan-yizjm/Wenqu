# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path

root = Path(SPECPATH)
a = Analysis(
    [str(root / "src" / "product_entry.py")],
    pathex=[str(root)],
    binaries=[],
    datas=[(str(root / "web" / "dist"), "web/dist"),
           (str(root / "docs" / "用户指南.md"), "resources/docs"),
           (str(root / "docs" / "RELEASE_NOTES_0.2.2.md"), "resources/docs"),
           (str(root / "docs" / "THIRD_PARTY_LICENSES.md"), "resources/docs"),
           (str(root / "examples" / "欢迎使用.md"), "resources/examples")],
    hiddenimports=["keyring.backends.Windows", "uvicorn.logging", "uvicorn.loops.auto",
                   "uvicorn.protocols.http.auto", "uvicorn.protocols.websockets.auto",
                   "uvicorn.lifespan.on", "multipart", "python_multipart"],
    hookspath=[], hooksconfig={}, runtime_hooks=[], excludes=[], noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(pyz, a.scripts, [], exclude_binaries=True, name="Wenqu",
          debug=False, bootloader_ignore_signals=False, strip=False, upx=True,
          console=False, disable_windowed_traceback=False)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=True,
               upx_exclude=[], name="Wenqu")

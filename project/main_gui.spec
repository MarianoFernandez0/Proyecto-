# -*- mode: python ; coding: utf-8 -*-
import os
block_cipher = None

octave_scripts = [(os.path.join('src/SpermTrackingProject', s), 'src/SpermTrackingProject')
                  for s in os.listdir('src/SpermTrackingProject')]
octave_kernel = [('/home/lu/Desktop/project/Proyecto-/project/src/oct2py/kernel.json', 'octave_kernel')]
oct2py_data = [('/home/lu/Desktop/project/Proyecto-/project/src/oct2py/_pyeval.m', 'oct2py')]

data = octave_scripts + octave_kernel + oct2py_data

a = Analysis(['main_gui.py'],
             pathex=['/home/lu/Desktop/project/Proyecto-/project'],
             binaries=[],
             datas=data,
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          [],
          exclude_binaries=True,
          name='main_gui',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          console=True,
          icon='src/icon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main_gui')

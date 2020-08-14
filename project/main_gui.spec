# -*- mode: python ; coding: utf-8 -*-
import os
block_cipher = None

octave_scripts = [(os.path.join('src/SpermTrackingProject', s), 'src/SpermTrackingProject')
                  for s in os.listdir('src/SpermTrackingProject')]
octave_kernel = [('/home/mariano/Projects/TDE/git/Proyecto-/project/src/oct2py/kernel.json', 'octave_kernel')]
oct2py_data = [('/home/mariano/Projects/TDE/git/Proyecto-/project/src/oct2py/_pyeval.m', 'oct2py')]

data = octave_scripts + octave_kernel + oct2py_data

a = Analysis(['main_gui.py'],
             pathex=['/home/mariano/Projects/TDE/git/Proyecto-/project'],
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
          console=True )
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               upx_exclude=[],
               name='main_gui')

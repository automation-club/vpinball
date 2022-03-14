# Visual Pinball

*An open source pinball table editor and simulator.*

This project was started by Randy Davis, open sourced in 2010 and continued by the Visual Pinball development team. This is the official repository.

## Automation Club Setup Instructions
1) Clone this repository and set it up to work with your IDE of choice. I reccomend using Visual Studio 2022, as that is what it's configured for.
2) Click [this link](https://github.com/vpinball/pinmame/releases/download/v3.4-336-cb9701e/Main.Download.-.VPinMAME34_Minimal.zip) to download the compressed folder for VPinMAME. Extract the folder, rename it to VPinMAME, and put it in your vpinball/ directory.
3) Navigate to the VPinMAME folder you just extracted amd launch Setup64.exe. Follow the default installation.
4) Download the [Bad Cats Table](https://www.vpforums.org/index.php?app=downloads&module=display&section=download&do=do_download&hash=0ad987d3b683c7b9c9db4b8bdbd9dc8a&id=77588) and extract it to vpinball/x64/Debug/Tables.
5) Download the [Bad Cats ROM](https://www.vpforums.org/index.php?app=downloads&module=display&section=download&do=do_download&hash=7317d227db5b5198c86a5b7f2b828535&id=15548) and move it to vpinball/VPinMAME/roms/ (you might have to create a roms/ directory within VPinMAME). **DO NOT DECOMPRESS THE ROMS FILE**
6) Compile the code, run the socket_server.py scrip which will automatically launch the table, and you should have a working Visual Pinball Bad Cats Table interfacing with Python. 
If you are trying to launch the Bad Cats table, you will need the roms file found here: https://www.vpforums.org/index.php?app=downloads&showfile=931


## Features

- Simulates pinball table physics and renders the table with DirectX
- Simple editor to (re-)create any kind of pinball table
- Table logic (and game rules) can be controlled via Visual Basic Script
- Over 950 real/unique pinball machines, plus over 350 original creations were rebuilt/designed using the Visual Pinball X editor (over 2000 if one counts all released tables, incl. MODs and different variants), and even more when including its predecessor versions (Visual Pinball 9.X)
- Emulation of real pinball machines via [Visual PinMAME](https://github.com/vpinball/pinmame) is possible via Visual Basic Script
- Supports configurable camera views (for example for correct display in virtual cabinets)
- Support for Stereo3D output and Tablet/Touch input

# Visual Pinball

*An open source pinball table editor and simulator.*

This project was started by Randy Davis, open sourced in 2010 and continued by the Visual Pinball development team. This is the official repository.
This is the LAHS Automation Club fork
## Automation Club Setup Instructions
1) Clone this repository and set it up to work with your IDE of choice. I reccomend using Visual Studio 2022, as that is what it's configured for.
2) Click [this link](https://github.com/vpinball/pinmame/releases/download/v3.4-336-cb9701e/Main.Download.-.VPinMAME34_Minimal.zip) to download the compressed folder for VPinMAME. Extract the folder, rename it to VPinMAME, and put it in your vpinball/ directory.
3) Navigate to the VPinMAME folder you just extracted amd launch Setup64.exe. Follow the default installation.
4) Download the [Bad Cats Table](https://drive.google.com/file/d/14720LJ2FxGfDSd7bel5x97ney5mURGKs/view?usp=sharing) and extract it to vpinball/x64/Debug/Tables.
5) Download the [Bad Cats ROM](https://drive.google.com/file/d/18Aouh4Xikq9UkjnP9975L49STgszuUSm/view?usp=sharing) and move it to vpinball/VPinMAME/roms/ (you might have to create a roms/ directory within VPinMAME). **DO NOT DECOMPRESS THE ROMS FILE**
6) If you don't have DirectX, install it from [here](https://www.microsoft.com/en-us/download/details.aspx?id=6812)
7) Compile the code, run the socket_server.py scrip which will automatically launch the table, and you should have a working Visual Pinball Bad Cats Table interfacing with Python. 



## Contact for more info
Github User: @arhanjain
Email: arhanj@uw.edu or arhan.j04@gmail.com

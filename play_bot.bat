@echo off
cd /d "%~dp0"

set "botname=Croy"
set "botfile=AI_logic.py"

set "SDL_VIDEO_WINDOW_POS=80,585"
set "consolexpos=5"
set "consoleypos=40"
set "consolewidth=138"
set "consoleheight=24"

wt -w new --pos %consolexpos%,%consoleypos% --size %consolewidth%,%consoleheight% --title %botname% -d . python reversi_client.py --player %botfile% --name %botname%

@echo off
rem New training run or resumes from the checkpoint if it exists.
cd /d "%~dp0"
python train_self_play.py
pause

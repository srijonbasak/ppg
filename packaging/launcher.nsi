Unicode True
!include "FileFunc.nsh"
Name "Possibility Lab"
OutFile "dist-windows\PossibilityLab\PossibilityLab.exe"
Icon "PossibilityLab.ico"
RequestExecutionLevel user
SilentInstall silent
AutoCloseWindow true
Section
 System::Call 'kernel32::SetEnvironmentVariableW(w "QTWEBENGINE_CHROMIUM_FLAGS", w "--disable-gpu") i.r0'
 ${GetParameters} $R0
 ExecWait '"$EXEDIR\Engine.exe" $R0' $R1
 SetErrorLevel $R1
SectionEnd
